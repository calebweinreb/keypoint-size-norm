from kpsn_test import visualize as viz

def plot(
    plot_name,
    fit,
    cfg,
    **kwargs):
    
    return {
        f"{plot_name}-loss": viz.fitting.em_loss(
            fit['loss_hist'], fit['mstep_losses'],
            mstep_relative = not cfg['mstep_abs']),
        f'{plot_name}-reports': viz.fitting.report_plots(fit['reports'])
    }
    


defaults = dict(
    mstep_abs = False
)