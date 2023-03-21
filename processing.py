import numpy as np

def autocorrelation(img: np.ndarray) -> np.ndarray:
    """Calcule l'autocorrélation via l'algorithme FFT, normalisé par le max.

    Parametres
    ----------
    img      : np.ndarray
               Image dont on souhaite calculer l'autocorrélation.

    Retour
    ------
    img_     : np.ndarray
               Image de l'autocorrélation avec spectre centré.
    """
    img_ = np.fft.fft2(img)
    img_ = np.power(np.abs(img_), 2)
    img_ = np.fft.ifft2(img_)
    img_ = np.abs(np.fft.fftshift(img_)/np.nanmax(img_))

    return img_



def cinfinity(img: np.ndarray) -> float:
    """Calcule la grandeur C_inf caractéristique du comportement à l'infini de l'image.
    
    Parametres
    ----------
    img       : np.ndarray
                Image dont on veut calculer le coefficient Cinfini
    
    Retour
    ------
    cinf_     : float
                Valeur du coefficient.
    """
    cinf_ = np.power(img.mean(), 2) /np.power(img, 2).mean()
    return cinf_



def length_vs_angle(auto: np.ndarray, cinf: float, angle: float) -> float:
    """Calcule la longueur au contour défini par Cinf depuis le centre en fonction de l'angle.

    Parametres
    ----------
    auto     : np.ndarray
               Signal autocorrélation interprété comme image carrée de taille N x N
    cinf     : float
               Valeur de Cinf, définissant la distance d'autocorrélation.
    angle    : float
               Valeur de l'angle à considérer, en radians.
    """

    N = auto.shape[0]

    while(angle < 0      ): angle = angle + 2*np.pi
    while(angle > 2*np.pi): angle = angle - 2*np.pi
    if    angle > np.pi   : angle = angle - np.pi

    j_ = np.linspace(0, N/2-1, N//2).astype(np.int32)     # liste des indexes des colonnes pour parcourir la moitié de l'image
    i_ = np.round(j_ * np.tan(angle)).astype(np.int32)    # liste des indexes des lignes pour que [i, j] soit intercepté la demi-droite d'angle theta

    j_ = j_[np.abs(i_) < N//2]                            # masque ne conservant que les valeurs de j_ telles que i_ ne sort pas de l'image
    i_ = i_[np.abs(i_) < N//2]                            # idem pour i_

    l  = np.sqrt( i_.max()**2 + j_.max()**2 )             # initialisation de la longueur de sorte que le rayon soit celui de l'image (cas d'un signal traversant, sans contour)

    if angle == np.pi/2:                                  # prolongement par continuité 
        lr = length_vs_angle(auto, cinf, angle-np.pi/32)
        ll = length_vs_angle(auto, cinf, angle+np.pi/32)
        l  = 0.5 * (lr + ll)
 
    elif angle < np.pi/2:
        for (i, j) in zip(i_, j_):
            if auto[N//2-i, N//2+j] > cinf:
                continue
            else:
                l = np.sqrt(i**2 + j**2)
                break

    else:
        for (i, j) in zip(i_, j_):
            if auto[N//2-i, N//2-j] > cinf:
                continue
            else:
                l = np.sqrt(i**2 + j**2)
                break

    return l



def radial_profile(auto: np.ndarray, angle: float) -> tuple:
    """Extrait le profil de l'autocorrélation dans la direction donnée par l'angle prescrit.

    Parametres
    ----------
    auto     : np.ndarray
               Signal autocorrélation interprété comme image carrée de taille N x N
    angle    : float
               Angle formé avec la demi-droite partant du centre de l'image et vers la droite

    Retour
    ------
    (r, a)   : tuple(np.ndarray, np.ndarray)
               Couple de vecteurs, `r` étant le vecteur des rayons et `a` le vecteur des valeurs de l'autocorrélation.
    """

    N = auto.shape[0]

    while(angle < 0      ): angle = angle + 2*np.pi
    while(angle > 2*np.pi): angle = angle - 2*np.pi
    if    angle > np.pi   : angle = angle - np.pi

    j_ = np.linspace(0, N/2-1, N//2).astype(np.int32)     # liste des indexes des colonnes pour parcourir la moitié de l'image
    if angle == np.pi/2:
        i_ = np.zeros_like(j_)
    else:
        i_ = np.round(j_ * np.tan(angle)).astype(np.int32)

    j_ = j_[np.abs(i_) < N//2]                            # masque ne conservant que les valeurs de j_ telles que i_ ne sort pas de l'image
    i_ = i_[np.abs(i_) < N//2]                            # idem pour i_

    r  = np.sqrt( np.power(i_, 2) + np.power(j_, 2) )
    a  = np.zeros_like(r)
    for k in range(len(r)):
        if angle < np.pi/2:
            a[k] = auto[N//2-i_[k], N//2+j_[k]]
        else:
            a[k] = auto[N//2-i_[k], N//2-j_[k]]

    return (r, a)
