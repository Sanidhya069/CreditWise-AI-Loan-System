import unittest

from llm_module.llm_advisor import generate_advice


class TestFinancialAdvisor(unittest.TestCase):

    def test_high_risk(self):
        advice = generate_advice(85, 25000, 500000, 550)
        self.assertIn("High Risk", advice)
        self.assertIn("Reduce loan amount", advice)

    def test_moderate_risk(self):
        advice = generate_advice(70, 50000, 300000, 680)
        self.assertIn("Moderate Risk", advice)

    def test_low_risk(self):
        advice = generate_advice(40, 80000, 200000, 780)
        self.assertIn("Low Risk", advice)

    def test_high_risk_boundary(self):
        advice = generate_advice(71, 30000, 400000, 600)
        self.assertIn("High Risk", advice)


if __name__ == "__main__":
    unittest.main()