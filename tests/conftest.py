import os
import tempfile

# Unit tests must not write into the repository's mesh library: point it at a
# throw-away directory for the whole session (drivers still exercise the
# generate -> save -> load path).
_LIB = tempfile.mkdtemp(prefix="thermalio_mesh_library_")
os.environ.setdefault("THERMALIO_MESH_LIBRARY", _LIB)
