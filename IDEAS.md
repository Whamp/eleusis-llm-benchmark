
# IDEAS FOR IMPROVEMENTS

## Priority
- system prompt ? differentiate reasoning and non-reasoning calls ?

## Later
- implement a seed in config.yaml to make the game reproducible for testing purposes (for both random in the game logic, and to be passed to the LLM client)
- Make sure the code is robust to various llm failures (timeouts, rate limits, invalid responses, etc.) so that it does not crash the game.
- Apparently "set(), reversed()" is not supported in python generated code (when rule guessing for instance).