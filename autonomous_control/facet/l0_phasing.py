import sys
import logging
from time import sleep
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('/usr/local/facet/tools/python/')
from F2_pytools.f2bsaBuffer import f2BeamSynchronousBuffer

from autonomous_control.facet.optimization_utils import restore_on_error

logger = logging.getLogger("l0_phasing")

@restore_on_error(context="l0_phasing")
def l0_phasing(
    env,
    *,
    k = 4,
    p0 = -15,
    pf = 15,
    Nshots = 100,
    makeplot=False,
    ):
    """
    Create and run the automatic L0 (10-4 and 10-8) phasing controller.

    Parameters
    ----------
    env : Any
        Badger environment
    k: int
        klystron ID, must be 4 or 8
    p0: numeric
        desired initial waveguide phase for scan
    pf: numeric
        desired finall waveguide phase for scan
    Nshots: int
        number of points of BSA data to acquire
    makeplot: bool, optional (default=False)
        flag to show phase scan result plot

    Returns
    -------
    None
    """
    scandata = fast_phase_scan_l0(env=env, k=k, p0=p0, pf=pf, Nshots=Nshots)
    scandata = fit_beam_phase(scandata)
    for k,v in scandata.items():
        logger.debug(f'{k}: {v}')
    if makeplot:
        plot_scan_result(scandata)
    if abs(scandata['psi_meas']) < 20.0:
        apply_correction = True
    else:
        apply_correction = input(f"Measured psi = {scandata['psi_meas']}. Update REFPOC? [y/n]").lower() == 'y'
    if apply_correction:
        correct_phase_error(env, scandata)


def fast_phase_scan_l0(env, k, p0, pf, Nshots):
    """ fast BSA RF phase scan for L0-A/B only """
    if k not in [4,8]:
        raise ValueError('klys must be 41 (L0A) or 81 (L0B)')

    fbeam_pv = 'EVNT:SYS1:1:BEAMRATE'
    lfb_control_pv = 'PHYS:SYS1:1:F2LFB_DL10E'
    sfb_control_pv = f'KLYS:LI10:{k}1:SFB_PDIS'
    sfb_pdes_pv = f'KLYS:LI10:{k}1:SFB_PDES'
    ffb_pdes_pv = f'KLYS:LI10:{k}1:PDES'
    refpoc_pv = f'KLYS:LI10:{k}1:REFPOC'

    buf = f2BeamSynchronousBuffer(
        EPICS_address_list=[
            f'ACCL:LI10:{k}1:W0C0:FAST_PACT', # cavity phase
            f'BPMS:IN10:731:X',               # DL10 energy BPM
            f'BPMS:IN10:731:TMIT',            # for data sanitization
            ],
        EPICS_Npts=Nshots,
        verbose=True,
        nowait=True
        )

    beam_rate = env.get_variables([fbeam_pv])[fbeam_pv]
    pdes_init = env.get_variables([ffb_pdes_pv])[ffb_pdes_pv]
    sfb_pdes_init = env.get_variables([sfb_pdes_pv])[sfb_pdes_pv]
    poc_init = env.get_variables([refpoc_pv])[refpoc_pv]

    # calculate FFB PDES scan range based on desired waveguide phase scan range
    poffset = pdes_init - sfb_pdes_init
    scan_range = np.linspace(p0 + poffset, pf + poffset, Nshots)

    # zero klystron phase, let RF/beam feedbacks converge, then disable them
    logger.info('zeroing pdes')
    env.set_variables({sfb_pdes_pv: 0})
    sleep(3.0)
    env.set_variables({
        sfb_control_pv: 0,
        lfb_control_pv: 0,
    })

    # set initial PDES and wait for FFB to settle
    logger.info('initializing scan')
    env.set_variables({ffb_pdes_pv: scan_range[0]})
    sleep(3.0)

    # fast scan -- 1 caput per shot
    for phi in scan_range:
        logger.info(f'setting ffb pdes = {phi: .3f}')
        env.set_variables({ffb_pdes_pv: phi})
        sleep(1/beam_rate)

    # get BPM and phase readback BSA data
    _, syncdata = buf.get_data()

    # re-enable feedbacks, restore initial phase settings
    logger.info('restoring initial settings')
    env.set_variables({
        sfb_pdes_pv: sfb_pdes_init,
        ffb_pdes_pv: pdes_init,
        sfb_control_pv: 1,
        lfb_control_pv: 1,
    })

    return {
        'klys': k,
        'p0': p0,
        'pf': pf,
        'Nshots': Nshots,
        'poffset': poffset,
        'phi': syncdata[0,:],
        'x': syncdata[1,:],
        'tmit': syncdata[2,:],
        'pdes_init': pdes_init,
        'poc_init': poc_init,
        'ts': datetime.now().strftime(f'%y%m%d%H%M%S'),
        }

def fit_beam_phase(scandata):
    """ fit phase scan data to x = Acos(phi) using OLS """
    phimeas = np.deg2rad(scandata['phi'])
    xmeas   = scandata['x']
    M_t     = np.vstack((np.cos(phimeas), np.sin(phimeas), np.ones(phimeas.shape)))
    M       = np.transpose(M_t)
    pinv    = np.linalg.inv(M_t @ M)
    a       = pinv @ M_t @ xmeas
    Ameas   = np.sign(a[0]) * np.sqrt(a[0]**2 + a[1]**2)
    psimeas = np.arcsin(np.deg2rad(a[1]/Ameas))
    scandata['psi_meas'] = _wrapto180(psimeas)
    scandata['poc_new'] = _wrapto180(scandata['poc_init'] + scandata['psi_meas'])
    scandata['A_meas'] = Ameas
    scandata['B_meas'] = a[2]
    return scandata

def correct_phase_error(env, scandata):
    """ adjust slow feedback REFPOC to correct measured phase error """
    addr = f"KLYS:LI10:{scandata['klys']}1:REFPOC"
    env.set_variables({addr: scandata['poc_new']})
    logger.info(f"{addr} updated: {scandata['poc_init']: .3f} --> {scandata['poc_new']: .3f}")

def plot_scan_result(scandata):
    """ generate phase scan plot, raw data + fit """
    plt.scatter(scandata['phi'], scandata['x'])
    _fit_phi = np.linspace(min(scandata['phi']), max(scandata['phi']), 200,)
    _fit_X = scandata['A_meas'] * np.cos(np.deg2rad(_fit_phi) - scandata['psi_meas']) + scandata['B_meas']
    plt.plot(_fit_phi, _fit_X, color='r')
    plt.xlabel('phi (degS)')
    plt.ylabel('X (mm)')
    plt.title(f"K10-{scandata['klys']} phase scan\nerr={scandata['psi_meas']: .3f}")
    plt.show()

def _wrapto180(deg): return (deg+180.) % 360. - 180.

