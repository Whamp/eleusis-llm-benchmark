
# IDEAS FOR IMPROVEMENTS

## Priority
- system prompt ? differentiate reasoning and non-reasoning calls ?

## Later
- implement a seed in config.yaml to make the game reproducible for testing purposes (for both random in the game logic, and to be passed to the LLM client)
- Make sure the code is robust to various llm failures (timeouts, rate limits, invalid responses, etc.) so that it does not crash the game.

## Complexity measurement on evaluate rules:

Example of code snippet for inspiration

```
import ast

def code_complexity(code: str) -> dict:
    """Return AST node count and cyclomatic complexity for Python code."""
    tree = ast.parse(code)
    
    node_count = 0
    cyclomatic = 1  # base complexity
    
    for node in ast.walk(tree):
        node_count += 1
        
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            cyclomatic += 1
        elif isinstance(node, ast.BoolOp):
            # each 'and'/'or' adds (n-1) decision points
            cyclomatic += len(node.values) - 1
    
    return {
        'node_count': node_count,
        'cyclomatic': cyclomatic
    }


# Test it
if __name__ == "__main__":
    rules = [
        ("always_true", "return True"),
        ("rank_higher", "if not mainline:\n    return True\nreturn card.rank > mainline[-1].rank"),
        ("same_suit_or_rank", "if not mainline:\n    return True\nlast = mainline[-1]\nreturn card.suit == last.suit or card.rank == last.rank"),
    ]
    
    for name, code in rules:
        result = code_complexity(code)
        print(f"{name:20} -> nodes: {result['node_count']:2}, cyclomatic: {result['cyclomatic']}")
```

Output would look something like:
```
always_true          -> nodes:  4, cyclomatic: 1
rank_higher          -> nodes: 18, cyclomatic: 2
same_suit_or_rank    -> nodes: 26, cyclomatic: 3
```
