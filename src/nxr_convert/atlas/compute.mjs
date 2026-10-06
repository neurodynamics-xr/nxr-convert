/**
 * nxr-atlas compute — the nxr-compute calls the atlas needs, run headless on one mesh.
 *
 *   node compute.mjs frames    --dir D   → D/field.f64 (trivial-connection face field, [nF×3]),
 *                                          D/ortho.f64, D/logmap.f64 ([nV×2], from the north pole),
 *                                          D/frames.json (χ, Gauss–Bonnet)
 *   node compute.mjs transport --dir D   → D/transport.f64: for each source s in D/sources.i32 with its
 *                                          tangent vector in D/vectors.f64 ([nS×3]), the vector heat
 *                                          method's parallel transport to every vertex, [nS×nV×3]
 *
 * Inputs: D/vertices.f64 [nV×3], D/faces.i32 [nF×3], D/args.json ({north, south} for frames).
 * The binding is nxr-compute's Node addon — the one the app pins (app/scripts/nxr-compute.release.json, fetched into
 * app/vendor/nxr-compute-addon/). NXR_COMPUTE is the directory holding its index.mjs (the release payload, or an
 * nxr-compute checkout's bindings/node); the Python side (frames.nxr_compute_dir) resolves it. Ported from nsp's
 * tools/nxr-atlas/compute.mjs (D135).
 */
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const mode = process.argv[2]
const dir = process.argv[process.argv.indexOf('--dir') + 1]
const rd = (f, T) => { const b = fs.readFileSync(path.join(dir, f)); return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT) }
const wr = (f, a) => fs.writeFileSync(path.join(dir, f), Buffer.from(a.buffer, a.byteOffset, a.byteLength))
const root = process.env.NXR_COMPUTE
if (!root) throw new Error('NXR_COMPUTE is not set (the directory holding the nxr-compute Node binding\'s index.mjs)')
// the release payload (index.mjs beside nxr_compute_addon.node) or a checkout (bindings/node/index.mjs, the addon at its root)
const flat = fs.existsSync(path.join(root, 'index.mjs'))
if (flat && !process.env.NXR_COMPUTE_ADDON_PATH && fs.existsSync(path.join(root, 'nxr_compute_addon.node')))
  process.env.NXR_COMPUTE_ADDON_PATH = path.join(root, 'nxr_compute_addon.node')
const { default: init } = await import(pathToFileURL(flat ? path.join(root, 'index.mjs') : path.join(root, 'bindings/node/index.mjs')).href)
const nxr = await init()
const m = nxr.createManifoldContext(Float64Array.from(rd('vertices.f64', Float64Array)), Int32Array.from(rd('faces.i32', Int32Array)))
const args = JSON.parse(fs.readFileSync(path.join(dir, 'args.json'), 'utf8'))

if (mode === 'frames') {
  const t = m.interpolate.trivial([args.north, args.south], [1, 1])
  wr('field.f64', Float64Array.from(t.directionVectors))
  wr('ortho.f64', Float64Array.from(t.orthogonalVectors))
  const lm = m.uv.logMap(args.north)
  wr('logmap.f64', Float64Array.from(lm.logCoords))
  fs.writeFileSync(path.join(dir, 'frames.json'), JSON.stringify({
    euler: t.eulerCharacteristic, gaussBonnet: t.gaussBonnetSatisfied, north: args.north, south: args.south }))
} else if (mode === 'transport') {
  const src = rd('sources.i32', Int32Array), vec = rd('vectors.f64', Float64Array)
  const nV = rd('vertices.f64', Float64Array).length / 3
  const out = new Float64Array(src.length * nV * 3)
  for (let i = 0; i < src.length; i++) {
    const r = m.interpolate.transport([src[i]], Array.from(vec.subarray(3 * i, 3 * i + 3)))
    out.set(r, i * nV * 3)
  }
  wr('transport.f64', out)
} else throw new Error(`mode ${mode}: frames | transport`)
