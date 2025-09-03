import ee
from geeo.level4.lsp import lsp

def run_level4(prm):
    """
    Run the level4 process with the given parameters.

    Args:
        params (dict): Dictionary obtained from level-3 processing.
    """

    # convert parameters to variables
    YEAR_MIN = prm.get('YEAR_MIN')
    YEAR_MAX = prm.get('YEAR_MAX')
    # LSP
    LSP = prm.get('LSP')
    LSP_BASE_IMGCOL = prm.get('LSP_BASE_IMGCOL')
    LSP_BAND = prm.get('LSP_BAND')
    LSP_YEAR_MIN = prm.get('LSP_YEAR_MIN')
    LSP_YEAR_MAX = prm.get('LSP_YEAR_MAX')
    LSP_ADJUST_SEASONAL = prm.get('LSP_ADJUST_SEASONAL')
    LSP_ADJUST_SEASONAL_MAX_DAYS = prm.get('LSP_ADJUST_SEASONAL_MAX_DAYS')

    # LAND SURFACE PHENOLOGY (LSP)
    if LSP:

        if LSP == 'POLAR':
        
            lsp_base_imgcol = prm.get(LSP_BASE_IMGCOL)
            # make sure imgcol exists
            if not lsp_base_imgcol:
                raise ValueError("LSP base ImageCollection not found.")

            if not LSP_YEAR_MIN or not LSP_YEAR_MAX:  # global
                LSP_YEAR_MIN = YEAR_MIN
                LSP_YEAR_MAX = YEAR_MAX

            # imgcol
            imgcol_lsp = lsp(
                imgcol=lsp_base_imgcol,
                band=LSP_BAND,
                year_min=LSP_YEAR_MIN,
                year_max=LSP_YEAR_MAX,
                adjust_seasonal=LSP_ADJUST_SEASONAL,
                adjust_seasonal_max_delta_days=LSP_ADJUST_SEASONAL_MAX_DAYS,
                return_imgcol=True
            )
            # img
            img_lsp = lsp(
                imgcol=lsp_base_imgcol,
                band=LSP_BAND,
                year_min=LSP_YEAR_MIN,
                year_max=LSP_YEAR_MAX,
                adjust_seasonal=LSP_ADJUST_SEASONAL,
                adjust_seasonal_max_delta_days=LSP_ADJUST_SEASONAL_MAX_DAYS,
                return_imgcol=False
            )
        else:
            raise ValueError(f"Unknown LSP method: {LSP}")

        prm['LSP'] = imgcol_lsp
        prm['LSP_IMG'] = img_lsp
    
    return prm

# EOF
