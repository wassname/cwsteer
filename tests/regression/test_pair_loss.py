"""Coupled-loss cancellation and production gradient checks. -- PI/OpenAI"""
import math
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from cwsteer.adapter import ModulatedLoRA, ModulatedPiSSA
from cwsteer.train import _KL_TOPK, _kl_topk_base, _odd_pair_loss, _per_sample_nll, train_step


torch.set_num_threads(1)


class PairLossTests(unittest.TestCase):
    def test_same_weight_for_all_four_terms_per_pair(self):
        nlls = [torch.tensor(x, dtype=torch.float64, requires_grad=True)
                for x in ([1., 4.], [3., 2.], [4., 2.], [2., 4.])]
        C = 0.7
        loss = _odd_pair_loss(*nlls, C=C)
        margin = C * ((nlls[2] - nlls[0]) - (nlls[3] - nlls[1]))
        weight = C * torch.sigmoid(-margin) / 2
        for grad, sign in zip(torch.autograd.grad(loss, nlls), (1, -1, -1, 1)):
            torch.testing.assert_close(grad, sign * weight)

    def test_reply_specific_even_terms_cancel_in_value_and_gradient(self):
        C = 1.3
        even = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
        odd = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
        cho_even = even[0] * C**2
        rej_even = even[1] * C**4
        nlls = (2 + cho_even - C * odd, 3 + rej_even + C * odd,
                2 + cho_even + C * odd, 3 + rej_even - C * odd)
        loss = _odd_pair_loss(*nlls, C=C)
        expected = torch.nn.functional.softplus(-4 * C**2 * odd).mean()
        torch.testing.assert_close(loss, expected)
        g_even, g_odd = torch.autograd.grad(loss, (even, odd))
        torch.testing.assert_close(g_even, torch.zeros_like(even), atol=0, rtol=0)
        self.assertTrue((g_odd < 0).all())

    def test_zero_strength_and_extreme_margins(self):
        nlls = [torch.tensor([0., 1000.], requires_grad=True) for _ in range(4)]
        loss = _odd_pair_loss(*nlls, C=0)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        for grad in torch.autograd.grad(loss, nlls):
            torch.testing.assert_close(grad, torch.zeros_like(grad))
        extreme = [torch.tensor(x, requires_grad=True)
                   for x in ([0., 1000.], [1000., 0.], [1000., 0.], [0., 1000.])]
        loss = _odd_pair_loss(*extreme, C=1)
        self.assertTrue(torch.isfinite(loss))
        for grad in torch.autograd.grad(loss, extreme):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertLessEqual(grad.abs().max().item(), 0.5)

    def test_train_step_matches_direct_coupled_gradient(self):
        for adapter_cls in (ModulatedLoRA, ModulatedPiSSA):
            for kl_lambda in (0., 0.03):
                with self.subTest(adapter=adapter_cls.__name__, kl=kl_lambda):
                    torch.manual_seed(7)
                    config = Qwen3Config(vocab_size=300, hidden_size=16,
                                          intermediate_size=32, num_hidden_layers=1,
                                          num_attention_heads=2, num_key_value_heads=1,
                                          head_dim=8, use_cache=False)
                    model = Qwen3ForCausalLM(config).eval()
                    kwargs = {"selection_score": "s_only"} if adapter_cls is ModulatedPiSSA else {}
                    adapter = adapter_cls(model, r=2, dtype=torch.float32, **kwargs)
                    params = list(adapter.parameters())
                    ip = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8]])
                    ir = torch.tensor([[1, 2, 9, 10, 11], [1, 2, 12, 13, 14]])
                    lp, lr = ip.clone(), ir.clone()
                    lp[:, :2] = lr[:, :2] = -100
                    attn = torch.ones_like(ip)
                    C = 0.7
                    base = []
                    with torch.no_grad(), adapter(model, c=0):
                        for ids in (ip, ir):
                            top = model(input_ids=ids, attention_mask=attn).logits.topk(_KL_TOPK, dim=-1)
                            base.append((top.values.log_softmax(-1), top.indices))
                    with adapter(model, c=C):
                        cp = model(input_ids=ip, attention_mask=attn).logits
                        rp = model(input_ids=ir, attention_mask=attn).logits
                    with adapter(model, c=-C):
                        cn = model(input_ids=ip, attention_mask=attn).logits
                        rn = model(input_ids=ir, attention_mask=attn).logits
                    nlls = (_per_sample_nll(cp, lp), _per_sample_nll(rp, lr),
                            _per_sample_nll(cn, lp), _per_sample_nll(rn, lr))
                    margin = C * ((nlls[2] - nlls[0]) - (nlls[3] - nlls[1]))
                    direct = torch.nn.functional.softplus(-margin).mean()
                    direct = direct + 0.5 * kl_lambda * (
                        _kl_topk_base(cp, *base[0], lp) + _kl_topk_base(rn, *base[1], lr))
                    expected = torch.autograd.grad(direct, params)
                    trace = train_step(model, adapter, ip, lp, attn, ir, lr, attn,
                                       params, C=C, kl_lambda=kl_lambda)
                    for p, grad in zip(params, expected):
                        torch.testing.assert_close(p.grad, grad, atol=2e-7, rtol=2e-4)
                    self.assertTrue(all(math.isfinite(v) for v in trace.values()))
                    self.assertGreater(trace["g_nll_norm"], 0)
                    self.assertEqual(adapter.c, 0)
                    self.assertFalse(adapter._attached)


if __name__ == "__main__":
    unittest.main()
