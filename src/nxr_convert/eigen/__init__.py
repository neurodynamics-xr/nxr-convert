"""THE EIGENMODE TOOLBOX — the Editor's own methods in Python, run on the STORE's surfaces (never a source format's), so a basis prepared
here is a basis of the very mesh the Editor holds (2026-10-26: nsp's basis, glued from Brainstorm's winding, did not match the Editor's).

    glue           ``glue.py``       the Editor's ``glueMesh`` (editor/src/editor/glue.ts): the hemispheres made one closed surface
    intrinsic      ``intrinsic.py``  its intrinsic Delaunay mesh (engine/toolbox/differential/intrinsicDelaunay.ts): mollify, flips, mass
    Hodge–Dirac    ``hodge_dirac.py`` its pencil (K, M) (…/hodgeDirac.ts) and the solve, written as nsp's split derivative for ``hodge``

Each is a line-for-line port held EQUAL to the TypeScript by ``tests/test_eigen_parity.py`` (golden files the TypeScript writes); a change on
one side fails there. Positions are rounded to float32 first, as the Editor receives them.
"""
