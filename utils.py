from typing import Dict, Callable
from os import environ
import git

def get_registry_decorator(registry: Dict) -> Callable:

    def register(name: str):

        def decorator(cls: Callable):

            assert (
                not name in registry
            ), f"No duplicate registry names. '{name}' was registerd more than once."

            registry[name] = cls

            return cls

        return decorator

    return register

def log_on_main(*args) -> None:
    """Prints a message if running on the global (including all nodes) main thread, otherwise does nothing. If there is no multiprocess it will print.

    Args:
        message (str): message to print.
    """
    if environ.get("RANK", "0") == "0":
        print(*args)

def get_git_info(repo_path="."):
    """
    Returns a tuple (branch_name, commit_hash) for the given Git repository path.
    If the repository or commands fail, returns (None, None).
    
    - repo_path: The path to the Git repo (default is current directory).
    """
    try:
        # Initialize a Repo object pointing to the given path
        repo = git.Repo(repo_path, search_parent_directories=True)

        # If HEAD is detached, 'active_branch' will raise an exception
        if repo.head.is_detached:
            branch_name = None
        else:
            branch_name = repo.active_branch.name
        
        commit_hash = repo.head.commit.hexsha

        return branch_name, commit_hash
    
    except Exception as e:
        # If it's not a Git repository or any error occurs
        print(f"Error getting Git info: {e}")
        return None, None

