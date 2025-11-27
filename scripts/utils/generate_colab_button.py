import os
import json

COLAB_BADGE_TEMPLATE = """
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path}
)
""".strip()

def has_colab_badge(cell):
    """Check if a markdown cell already contains the Colab badge."""
    if cell["cell_type"] != "markdown":
        return False
    return "colab.research.google.com" in "".join(cell["source"]).lower()

def add_badge_to_notebook(nb_path, github_user, repo, branch):
    """Add Colab badge to a single notebook file (.ipynb)."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    rel_path = os.path.relpath(nb_path).replace("\\", "/")

    badge = COLAB_BADGE_TEMPLATE.format(
        user=github_user,
        repo=repo,
        branch=branch,
        path=rel_path
    )

    if not cells:
        nb["cells"] = [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [badge + "\n\n"]
        }]
    else:
        first_cell = cells[0]
        if not has_colab_badge(first_cell):
            if first_cell["cell_type"] != "markdown":
                nb["cells"].insert(0, {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [badge + "\n\n"]
                })
            else:
                first_cell["source"] = [badge + "\n\n"] + first_cell["source"]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


def add_badge_to_all_notebooks(root_dir, github_user, repo, branch):
    """
    Walk through directories under root_dir and add a Colab badge
    to all Jupyter notebooks (.ipynb).
    """
    for folder, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".ipynb"):
                nb_path = os.path.join(folder, fname)
                add_badge_to_notebook(nb_path, github_user, repo, branch)
