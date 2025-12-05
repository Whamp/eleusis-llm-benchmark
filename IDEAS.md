
# IDEAS FOR IMPROVEMENTS

## Priority
- system prompt ?

## Later
- implement a seed in config.yaml to make the game reproducible for testing purposes (for both random in the game logic, and to be passed to the LLM client)
- implement a total token count in each client so we monitor usage and costs (print them at the end of each round), and response time count
- For each player, gather stats on average confidence level, average confidence level when making a guess, when correct, etc.
- Make sure the code is robust to various llm failures (timeouts, rate limits, invalid responses, etc.) so that it does not crash the game.
- Apparently "set()" is not supported in python generated code (when rule guessing for instance). 
- Abstract the game client interface so that we can easily plug in different LLM providers (OpenAI, Anthropic).
- Make sure tournament results are saved after each round in a JSON file to avoid losing data in case of crash.
- what happens at zero cards ?