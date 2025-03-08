# File: /sgwb3d/sgwb3d/gw3d.py

import numpy as np
from scipy.special import jv, legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt

class GW3D:
    """
    Compute angular power spectra and correlation functions for gravitational waves
    in 3D space with different polarization modes
    """
    
    def __init__(self, l_max=20):
        """
        Initialize with maximum multipole moment to compute
        
        Parameters:
        -----------
        l_max : int
            Maximum multipole moment
        """
        self.l_max = l_max
        self.ell = np.arange(0, l_max+1)
        
    def _R_L_tensor(self, kr, v):
        """Radial function for tensor mode longitudinal component"""
        vph = 1/v  # Phase velocity
        return -jv(2, kr) / (kr)**2
        
    def _R_L_vector(self, kr, v):
        """Radial function for vector mode longitudinal component"""
        vph = 1/v  # Phase velocity
        return -jv(1, kr) / kr
        
    def _R_L_ST(self, kr, v):
        """Radial function for scalar-transverse mode longitudinal component"""
        vph = 1/v  # Phase velocity
        return -jv(0, kr)
        
    def _R_L_SL(self, kr, v):
        """Radial function for scalar-longitudinal mode longitudinal component"""
        vph = 1/v  # Phase velocity
        return -3*jv(0, kr)/(kr)**2 + 3*jv(1, kr)/(kr)**3
    
    def _R_E_tensor(self, kr, v):
        """Radial function for tensor mode E component"""
        return jv(2, kr)
    
    def _R_E_vector(self, kr, v):
        """Radial function for vector mode E component"""
        return jv(1, kr)
    
    def _R_E_ST(self, kr, v):
        """Radial function for scalar-transverse mode E component"""
        return jv(0, kr)
    
    def _R_E_SL(self, kr, v):
        """Radial function for scalar-longitudinal mode E component"""
        return -jv(0, kr) + 2*jv(1, kr)/kr
    
    def _R_B_tensor(self, kr, v):
        """Radial function for tensor mode B component"""
        return jv(2, kr)
    
    def _R_B_vector(self, kr, v):
        """Radial function for vector mode B component"""
        return jv(1, kr)
    
    def _R_B_ST(self, kr, v):
        """Radial function for scalar-transverse mode B component"""
        return 0
    
    def _R_B_SL(self, kr, v):
        """Radial function for scalar-longitudinal mode B component"""
        return 0
        
    def _F_z(self, ell, alpha, v):
        """
        Projection factor for PTA observable
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        vph = 1/v  # Phase velocity
        
        if alpha == 'tensor':
            if ell == 2:
                return -1j * vph / 2 * (1/3)
            elif ell > 2:
                return -1j * vph / 2 * (2/((ell-1)*ell*(ell+1)*(ell+2)))
            else:
                return 0
                
        elif alpha == 'vector':
            if ell == 1:
                return -1j * vph / 2 * (1/2)
            elif ell == 2:
                return -1j * vph / 2 * (1/10)
            elif ell > 2:
                return -1j * vph / 2 * (1/((ell-1)*(ell+2)))
            else:
                return 0
                
        elif alpha == 'ST':
            if ell == 0:
                return -1j * vph / 2 * (1/3)
            elif ell == 1:
                return -1j * vph / 2 * (1/3)
            else:
                return -1j * vph / 2 * (1-v**2)/np.sqrt(2) * (1/3) * (-1 if ell == 2 else 0)
                
        elif alpha == 'SL':
            if ell == 0:
                return -1j * vph / 2 * (1/3)
            elif ell == 1:
                return 0
            elif ell == 2:
                return -1j * vph / 2 * (1/3) * np.sqrt(2)/(1-v**2)
            else:
                return 0
        
        return 0
    
    def _F_E(self, ell, alpha, v):
        """
        Projection factor for astrometry E-mode
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        if alpha == 'tensor':
            if ell == 2:
                return 1/2
            elif ell > 2:
                return np.sqrt((ell+2)*(ell-1)/((ell+1)*ell)) / (ell*(ell+1))
            else:
                return 0
                
        elif alpha == 'vector':
            if ell == 1:
                return 1/2
            elif ell >= 2:
                return np.sqrt(ell/(ell+1)) / (ell*(ell+1))
            else:
                return 0
                
        elif alpha == 'ST':
            if ell == 1:
                return 1/2
            elif ell >= 2:
                return (1-v**2)/np.sqrt(2) * (1/(ell*(ell+1))) * (-1 if ell == 2 else 0)
            else:
                return 0
                
        elif alpha == 'SL':
            if ell == 1:
                return 1/2
            elif ell == 2:
                return np.sqrt(2)/(1-v**2) * (1/(ell*(ell+1)))
            elif ell > 2:
                return 0
            else:
                return 0
        
        return 0
    
    def _F_B(self, ell, alpha, v):
        """
        Projection factor for astrometry B-mode
        
        Parameters:
        -----------
        ell : int
            Multipole moment
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        if alpha == 'tensor':
            if ell >= 2:
                return np.sqrt((ell+2)*(ell-1)/((ell+1)*ell)) / (ell*(ell+1))
            else:
                return 0
                
        elif alpha == 'vector':
            if ell >= 1:
                return np.sqrt(ell/(ell+1)) / (ell*(ell+1))
            else:
                return 0
                
        elif alpha == 'ST' or alpha == 'SL':
            # Scalar modes don't contribute to B-mode
            return 0
        
        return 0

    def C_zz(self, ell, alpha, v):
        """
        Compute pulsar timing array power spectrum
        
        Parameters:
        -----------
        ell : int or array
            Multipole moment(s)
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        if np.isscalar(ell):
            Fz = self._F_z(ell, alpha, v)
            return 32*np.pi**2 * abs(Fz)**2
        else:
            return np.array([self.C_zz(l, alpha, v) for l in ell])

    def C_EE(self, ell, alpha, v):
        """
        Compute astrometry E-mode power spectrum
        
        Parameters:
        -----------
        ell : int or array
            Multipole moment(s)
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        if np.isscalar(ell):
            FE = self._F_E(ell, alpha, v)
            return 32*np.pi**2 * abs(FE)**2
        else:
            return np.array([self.C_EE(l, alpha, v) for l in ell])

    def C_BB(self, ell, alpha, v):
        """
        Compute astrometry B-mode power spectrum
        
        Parameters:
        -----------
        ell : int or array
            Multipole moment(s)
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        if np.isscalar(ell):
            FB = self._F_B(ell, alpha, v)
            return 32*np.pi**2 * abs(FB)**2
        else:
            return np.array([self.C_BB(l, alpha, v) for l in ell])

    def C_zE(self, ell, alpha, v):
        """
        Compute cross-correlation power spectrum
        
        Parameters:
        -----------
        ell : int or array
            Multipole moment(s)
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        """
        if np.isscalar(ell):
            Fz = self._F_z(ell, alpha, v)
            FE = self._F_E(ell, alpha, v)
            return 32*np.pi**2 * (Fz.conjugate() * FE).real
        else:
            return np.array([self.C_zE(l, alpha, v) for l in ell])
    
    def normalize_spectrum(self, C_l, alpha):
        """Normalize spectrum by quadrupole (ell=2) contribution"""
        if alpha == 'ST' and np.isclose(C_l[2], 0):
            # ST mode special case - normalize by max value
            return C_l / np.max(C_l[C_l > 0]) if np.any(C_l > 0) else C_l
        else:
            return C_l / C_l[2] if C_l[2] > 0 else C_l
            
    def correlation_function(self, alpha, v, observable='zz', angles=None):
        """
        Compute correlation function C(θ) from power spectrum
        
        Parameters:
        -----------
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        v : float
            Group velocity (0 < v ≤ 1)
        observable : str
            'zz', 'EE', 'BB', or 'zE'
        angles : array, optional
            Angular separations in degrees
            
        Returns:
        --------
        angles : array
            Angular separations in degrees
        corr : array
            Correlation function values
        """
        if angles is None:
            angles = np.linspace(0, 180, 181)
        
        # Get power spectrum
        if observable == 'zz':
            C_l = self.C_zz(self.ell, alpha, v)
        elif observable == 'EE':
            C_l = self.C_EE(self.ell, alpha, v)
        elif observable == 'BB':
            C_l = self.C_BB(self.ell, alpha, v)
        elif observable == 'zE':
            C_l = self.C_zE(self.ell, alpha, v)
        
        # Normalize
        C_l = self.normalize_spectrum(C_l, alpha)
        
        # Calculate correlation function
        corr = np.zeros_like(angles, dtype=float)
        cos_theta = np.cos(np.radians(angles))
        
        for i, ct in enumerate(cos_theta):
            for l in range(len(self.ell)):
                if C_l[l] != 0:
                    leg = legendre(l)
                    corr[i] += (2*l + 1) / (4*np.pi) * C_l[l] * leg(ct)
        
        return angles, corr

    def plot_power_spectrum(self, alpha, velocities, observable='zz', normalize=True, logscale=False):
        """
        Plot power spectrum for different velocities
        
        Parameters:
        -----------
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        velocities : list
            List of group velocities to plot
        observable : str
            'zz', 'EE', 'BB', or 'zE'
        normalize : bool
            Whether to normalize by quadrupole
        logscale : bool
            Whether to use log scale for y-axis
        """
        plt.figure(figsize=(10, 6))
        
        for v in velocities:
            if observable == 'zz':
                C_l = self.C_zz(self.ell, alpha, v)
                title = f"$C_\\ell^{{zz}}$ Power Spectrum - {alpha} mode"
            elif observable == 'EE':
                C_l = self.C_EE(self.ell, alpha, v)
                title = f"$C_\\ell^{{EE}}$ Power Spectrum - {alpha} mode"
            elif observable == 'BB':
                C_l = self.C_BB(self.ell, alpha, v)
                title = f"$C_\\ell^{{BB}}$ Power Spectrum - {alpha} mode"
            elif observable == 'zE':
                C_l = self.C_zE(self.ell, alpha, v)
                title = f"$C_\\ell^{{zE}}$ Power Spectrum - {alpha} mode"
            
            if normalize:
                C_l = self.normalize_spectrum(C_l, alpha)
                
            plt.plot(self.ell, C_l, 'o-', label=f'v = {v}')
        
        plt.xlabel('Multipole $\ell$')
        plt.ylabel('Normalized Power' if normalize else 'Power')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if logscale:
            plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_correlation(self, alpha, velocities, observable='zz'):
        """
        Plot correlation function for different velocities
        
        Parameters:
        -----------
        alpha : str
            Polarization mode ('tensor', 'vector', 'ST', 'SL')
        velocities : list
            List of group velocities to plot
        observable : str
            'zz', 'EE', 'BB', or 'zE'
        """
        plt.figure(figsize=(10, 6))
        
        # Add Hellings-Downs curve for reference if PTA
        if observable == 'zz':
            angles = np.linspace(0, 180, 181)
            cos_theta = np.cos(np.radians(angles))
            hd_curve = 0.5 + 0.5*cos_theta + (1-cos_theta)*np.log((1-cos_theta)/2)/2
            plt.plot(angles, hd_curve, 'k--', label='Hellings-Downs')
            
        for v in velocities:
            angles, corr = self.correlation_function(alpha, v, observable)
            plt.plot(angles, corr, label=f'v = {v}')
            
        plt.xlabel('Angular separation (degrees)')
        plt.ylabel('Correlation')
        if observable == 'zz':
            title = f"Pulsar timing correlation - {alpha} mode"
        elif observable == 'EE':
            title = f"Astrometry E-mode correlation - {alpha} mode"
        elif observable == 'BB':
            title = f"Astrometry B-mode correlation - {alpha} mode"
        elif observable == 'zE':
            title = f"Cross correlation - {alpha} mode"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()