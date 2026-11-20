# Changelog

All notable changes to geeo are documented here.

## [1.1.0] - 2026-11-20

### Added
- **Spatial tiling:** new `run_tiled()` (`geeo/misc/tiling.py`). `ROI_TILES` now accepts `true`, a preset (e.g. `'EU_300'`) or a grid vector/asset; tile ID column is inferred for presets.
- **`plot_tsi()`** for visualising observations, interpolated series, data density and valid-observation counts.
- **Test suite** for product correctness (NVO, STM/folding, TSI, PBC, LSP, tiling) on synthetic images.
- **Documentation and examples:** HTML/PDF documentation, parameter files and geeo/vanilla scripts for the three paper examples, reworked habitat tutorial, updated workflow figure, `Makefile`.

### Changed
- **Documentation**, overhauled the worked examples (`geeo/docs/tutorials`), in particular `docs/tutorials/habitat-type-mapping_geeo.ipynb`
- **Unmixing (UMX) moved from Level-2 to Level-3**, with new `UMX_BASE_IMGCOL` setting. `run_level2()` no longer unmixes.
- **TSI** now spans the user-specified time range (`YEAR_MIN/MAX` or `DATE_MIN/MAX`) rather than first/last image, so interpolated time axes can differ from 1.0.
- **RBF interpolation** reworked for efficiency and adressed the an impact of artifacts because of the mask when bands differ in their native resolution (e.g., Sentinel-2).
- **Performance:** removed costly sorting after sensor merging (exports sort explicitly);

### Fixed
- Disabling masking and no resulting mask band present no longer breaks downstream steps (a constant valid `mask` band is added in this case).

## [1.0] - 2024-11-19

- Initial release.
