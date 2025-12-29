# RL training for Contextual Integrity

We use `verl` for RL training (https://github.com/volcengine/verl) with only small modifications e.g. reward function.

To run our training, first clone `verl` and follow the instructions to set up an environment.
Then, apply the changes in `changes.diff` to the `verl` repository to add the reward function and other modifications.

```bash
git clone https://github.com/volcengine/verl.git verl
cd verl
git apply ../changes.diff
```

Finally run

```bash
run_rl_for_contextual_integrity.sh
```

This will start the training process. The training process will take a while, so be patient. You can monitor the training progress in the terminal.
