# docs/gen_api.py
import importlib.util
import pkgutil

from mkdocs_gen_files import open as gen_open


def get_submodules(pkg, ref_dir):
    """Return a list of (module, md_path) for all submodules of pkg (excluding init)."""
    # Find the package's path
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        return []
    pkg_path = spec.submodule_search_locations[0]
    result = []
    for _, modname, _ in pkgutil.iter_modules([pkg_path]):
        if modname.startswith("__init__"):
            continue
        mod_full = f"{pkg}.{modname}"
        md_path = f"{ref_dir}/{modname}.md"
        result.append((mod_full, md_path))
    return result


SUBMODULES = [
    ("laplax.enums", "reference/enums.md"),
    ("laplax.register", "reference/register.md"),
]

# Dynamically add submodules for curv, eval, util
for pkg, ref_dir in [
    ("laplax.curv", "reference/curv"),
    ("laplax.eval", "reference/eval"),
    ("laplax.util", "reference/util"),
]:
    SUBMODULES.extend(get_submodules(pkg, ref_dir))

for mod, md_path in SUBMODULES:
    with gen_open(md_path, "w") as f:
        f.write(f"::: {mod}\n")
        f.write("\n")
