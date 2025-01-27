# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Google Gemini API Language Model."""

from collections.abc import Collection, Sequence
import copy
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from concordia.utils import text
import time

from typing_extensions import override

MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
DEFAULT_HISTORY = [
    {'role': 'user', 'parts': ['Continue my sentences. Never repeat their starts.']},
    {'role': 'model', 'parts': ['I always continue user-provided text and never repeat what the user already said.']},
    {'role': 'user', 'parts': ["Question: Is Jake a turtle?\nAnswer: Jake is "]},
    {'role': 'model', 'parts': ['not a turtle.']},
    {'role': 'user', 'parts': ["Question: What is Priya doing right now?\nAnswer: Priya is currently "]},
    {'role': 'model', 'parts': ['sleeping.']},
]

class GeminiLanguageModel(language_model.LanguageModel):
  """Language model via the Gemini API."""

  def __init__(
      self,
      api_key: str,
      model_name: str = 'gemini-pro',
      *,
      harm_block_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      sleep_periodically: bool = False,
  ) -> None:
    """Initializes a model instance using the Gemini API.

    Args:
      api_key: Gemini API key from Google AI Studio
      model_name: which language model to use
      harm_block_threshold: Safety threshold. Choose from {BLOCK_ONLY_HIGH,
        BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE, BLOCK_NONE}
      measurements: The measurements object to log usage statistics to
      channel: The channel to write the statistics to
      sleep_periodically: Whether to sleep between API calls to avoid rate limit
    """
    genai.configure(api_key=api_key)
    self._model = genai.GenerativeModel(model_name)
    self._measurements = measurements
    self._channel = channel
    self._safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: harm_block_threshold,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: harm_block_threshold,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: harm_block_threshold,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: harm_block_threshold,
    }
    self._sleep_periodically = sleep_periodically
    self._calls_between_sleeping = 10
    self._n_calls = 0

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del timeout
    if seed is not None:
      raise NotImplementedError('Seed not supported by Gemini API')
    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0):
      print('Sleeping for 10 seconds...')
      time.sleep(10)
    time.sleep(10)  

    chat = self._model.start_chat(history=copy.deepcopy(DEFAULT_HISTORY))
    response = chat.send_message(
        prompt,
        generation_config={
            'temperature': temperature,
            'max_output_tokens': max_tokens,
            'stop_sequences': terminators,
            'candidate_count': 1,
        },
        safety_settings=self._safety_settings
    )

    try:
      response_text = response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError) as e:
      print(f'Error parsing response: {e}')
      print(f'Prompt: {prompt}')
      print(f'Full response: {response}')
      response_text = ''

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(response_text)})
    return text.truncate(response_text, delimiters=terminators)

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    # ... (keep existing sample_choice implementation unchanged) ...
    sample = ''
    answer = ''
    for attempts in range(MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, MAX_MULTIPLE_CHOICE_ATTEMPTS)

      question = (
          'The following is a multiple choice question. Respond ' +
          'with one of the possible choices, such as (a) or (b). ' +
          f'Do not include reasoning.\n{prompt}')
      sample = self.sample_text(
          question,
          max_tokens=256,  # This is wasteful, but Gemini blocks lower values.
          temperature=temperature,
          seed=seed,
      )
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        print(f'Sample choice fail: {answer} extracted from {sample}.')
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel,
              {'choices_calls': attempts})
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError(
        (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
         f'extracted: {answer}')
    )