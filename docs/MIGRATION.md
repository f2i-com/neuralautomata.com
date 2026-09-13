# Separation from Zipp

The native NCA UI, fixed Node/Python runner, smoke tests and recordings moved from
Zipp's playground into this independent repository at the user's request.
The source baseline was Zipp test-repository commit `34919bae`; the move also includes
JSON content-type parameter support and renderer improvements (per-snapshot ring RMS
and texture storage reuse). No runtime dependency on Zipp remains.

The supplied research package is copied byte-for-byte under `research/fast_memory_language/`,
excluding generated Python caches. `research-import.json` records SHA-256 for every
imported file. Original research provenance, manifests and CPU results remain there.
They are historical records, not new GPU benchmarks. New local runs belong in `target/`.

The root Apache-2.0 LICENSE accompanies the code moved from Zipp. The research
package retains its original accompanying notices and provenance.
