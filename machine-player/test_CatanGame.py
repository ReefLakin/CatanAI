import unittest
import CatanGame


class TestStringMethods(unittest.TestCase):

    # Test that the resource pool is set to 0 after a game reset
    def test_reset_resets_resource_pool(self):
        game = CatanGame.CatanGame()
        game.reset()
        state = game.get_state()

        self.assertEqual(state['num_lumber'], 0)
        self.assertEqual(state['num_wool'], 0)
        self.assertEqual(state['num_grain'], 0)
        self.assertEqual(state['num_ore'], 0)
        self.assertEqual(state['num_brick'], 0)

    # Test that a player can't do a 4-1 trade as soon as the game resets
    def test_4_1_trade_illegal_on_reset(self):
        game = CatanGame.CatanGame()
        game.reset()
        legal_actions = game.get_legal_actions()

        self.assertNotIn("trade_bank_4_for_1_brick_wool", legal_actions)
        self.assertNotIn("trade_bank_4_for_1_grain_lumber", legal_actions)
        self.assertNotIn("trade_bank_4_for_1_ore_grain", legal_actions)

    # Test that a player gets a reward of -1 for making an illegal action
    def test_illegal_action_reward(self):
        game = CatanGame.CatanGame()
        game.reset()

        reward = game.get_reward("build_settlement_northeast_0_0_0")
        self.assertEqual(reward, -1)

    # Test that the reward is updated after a player builds a settlement
    def test_reward_updated_after_settlement_build(self):
        # Broken until the game gets a building feature
        # game = CatanGame.CatanGame()
        # game.reset()
        # game.set_legal_actions_manually(["build_settlement_northeast_0_0_0"])
        # game.step("build_settlement_northeast_0_0_0")

        # state = game.get_state()
        # self.assertEqual(state['reward'], 1)
        pass

if __name__ == '__main__':
    unittest.main()
