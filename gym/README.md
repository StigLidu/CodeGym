# CodeGym Systhesis Pipeline

We provide a sample coding problem dataset (`example/raw_problems.jsonl`) and here is the step-by-step CodeGym generation process:

**Step 1:** Run `gym/0_gym_gen.py` to generate prompts for environment synthesis. Note that if `--online-inference` is enabled, it will automatically call the OpenAI API to generate the environments, otherwise you should run inference on the prompts with your own LLMs.

**Step 2:** Process the generated environments by running `gym/0_gym_lease.py`. This will generate an environment jsonl file containing environments that have passed the compilation check.

**Step 3:** Run `gym/1_unit_test_gen.py` to generate unit tests for the environments. This will create comprehensive test cases to validate environment functionality and correctness. Similarly, if `--online-inference` is enabled, it will automatically call the OpenAI API to generate the unit tests. After running inference on the prompts, run `gym/1_unit_test_lease.py` to extract properly formatted unit test inputs.

**Step 4:** Run `gym/2_solve_fc_gen.py` to generate solution functions for the environments. This will create reference solutions that demonstrate how to use the provided tools to solve each environment. Similarly, if `--online-inference` is enabled, it will automatically call the OpenAI API to generate the solution functions. After running inference on the prompts, run `gym/2_solve_fc_lease.py` to extract properly formatted solution functions.

**Step 5:** Run `gym/3_solve_fc_with_unit_test.py` to check whether a solution function exists that passes all unit tests. If one exists, the environment is valid.

**Step 6:** Run `gym/4_extract_env_to_server.py` to produce all verified environments and corresponding task configurations.

