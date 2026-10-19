# Dataset construction

## Paper configuration

The default public configuration follows the manuscript tables:

- patch size: `64 x 64`;
- overlap ratio: `0.5` (stride `32`);
- pressure-corrector samples, rather than pairs of adjacent physical time steps;
- input channels, in order:
  `alpha, rhs, rAU, p_in, p_rgh_in, invDx2, invDy2, x, y, mask`;
- target: `delta_p_rgh`;
- physical-field padding: `symmetric`;
- coordinate-like channels (`x`, `y`, `invDx2`, `invDy2`): edge padding;
- mask: zero padding.

The generated arrays are `float32`. Geometry channels are stored in their raw
dataset representation and are not normalized by this pipeline.

## 1. Build pressure-corrector patches

```powershell
python dataset_pipeline_dp_rgh_new.py `
  --dataset-root "D:\path\to\openfoam-cases" `
  --out-root "D:\path\to\dp-rgh-patches"
```

Each OpenFOAM case is expected to contain `system/blockMeshDict`, numeric time
directories, the base fields `alpha.water`, `p`, and `p_rgh`, and corrector dumps
named `rhs_corrK`, `rAU_corrK`, and `p_rgh_corrK`.

The output layout is:

```text
out-root/
  dataset_manifest.json
  <case>/
    processed_corrector/<time>.npz
    patch_t0000.npz
    patch_t0001.npz
```

Patch files contain `X`, `Y`, `coords`, and `meta_json`. Existing patch files are
not overwritten unless `--force-patch` is supplied. Use `--help` to see all
options. If case directories are named `Case1`, `Case2`, and so on, pass
`--case-name-regex "^Case\d+$"`; the default accepts numeric names.

## 2. Validate the generated dataset

```powershell
python sanity_check.py `
  --patch-root "D:\path\to\dp-rgh-patches" `
  --report-json "D:\path\to\sanity-report.json"
```

The command returns a non-zero exit status when a patch is malformed or no patch
files are found. It checks shapes, finite values, coordinates, corrector chaining,
and approximate pressure-equation consistency.

## 3. Build aligned velocity supervision (optional)

```powershell
python dataset_pipeline_u_patch.py `
  --dataset-root "D:\path\to\openfoam-cases" `
  --dp-patch-root "D:\path\to\dp-rgh-patches" `
  --out-root "D:\path\to\u-patches"
```

By default, only rows from the final pressure corrector are retained. The script
reads the pressure patch metadata and uses exactly the same padding and crop
coordinates for `U`.

