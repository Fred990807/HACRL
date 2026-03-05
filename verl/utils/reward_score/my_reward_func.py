# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from verl.utils.reward_score  import gsm8k, math, prime_math, prime_code

# from verl.eval_math_rule.evaluation.parser import extract_answer, parse_ground_truth, strip_string

# TODO replace from . with from verl.utils.reward_score 
# the return should be a value
def my_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    # from verl.utils.reward_score  import math
        # res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:


    from verl.utils.reward_score  import math_verify
    res = math_verify.compute_score(solution_str, ground_truth)
    
    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])



def other_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None):
    import logging
    logging.getLogger('latex2sympy2_extended.math_normalization').setLevel(logging.ERROR)
    """Reward function that checks if the completion is the same as the ground truth."""
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
    )
    if len(gold_parsed) == 0 and '$' not in ground_truth:
        gold_parsed = parse(f'$${ground_truth}$$',
                            extraction_mode='first_match')
    if len(gold_parsed) == 0:
        gold_parsed = parse(r'\boxed{' + ground_truth + r'}',
                            extraction_mode='first_match')
    if len(gold_parsed) != 0:
        solution = solution_str
            
        # only perform latex match, first try answer in boxed, then try general match (must before a anchor sentence like answer is)
        # if an answer is put in a \boxed{}, whether it is surrounded by $$ does not make influence
        answer_parsed = parse(
            solution,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Compute binary rewards if verifiable, `None` otherwise to skip this example
        try:
            reward = float(verify(gold_parsed, answer_parsed))
        except Exception as e:
            reward = 0.0
    else:
        reward = 0.0

    return reward