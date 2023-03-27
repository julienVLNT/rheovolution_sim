import adelio            as io
import matplotlib.pyplot as plt
import matplotlib.tri    as tri
import numpy             as np
import os
import processing        as pro
import scipy.interpolate as int
import scipy.stats       as sta

print()

# Création du répertoire d'export
os.makedirs(os.path.join("analyse", "uniques"), exist_ok=True)
os.makedirs(os.path.join("analyse", "groupes"), exist_ok=True)

# Création d'un dictionnaire simulations.
simulations = {}

simulations["paths"] = []    # liste des chemins vers les dossiers des simulations
simulations["sim"]   = []    # liste des simulations individuelles, encore des dictionnaires
simulations["col"]   = []    # liste des couleurs pour l'affichage


# Lecture du fichier des chemins absolus vers les dossiers contenant les fichiers T et P définissant le groupe
# Le fichier est à la racine du programme et se nomme `chemins.txt``
# La première ligne permet de nommer le groupe.
with open("chemins.txt", "r") as file:
    lines = file.readlines()
    name  = lines[0]
    simulations["name"] = name if not name.endswith("\n") else name[:-1]
    # os.makedirs(os.path.join("analyse", "groupes", simulations["name"]), exist_ok=True)
    for line in lines[1:]:
        if line[-1] == "\n": line = line[:-1]    # Retire le saut de ligne `\n`
        if line != "":
            simulations["paths"].append(line)
            os.makedirs(os.path.join("analyse", "uniques", os.path.basename(line)), exist_ok=True)


# Pour chaque chemin renseigné...
for path in simulations["paths"]:

    print(f"> {path}")
    print(f"> " + "-"*len(path))

    # Créer un dictionnaire `simulation` contenant les données des champs
    simulation = {}

    simulation["name"] = os.path.basename(path)

    # Déclaration de l'objet Tfile et création de la liste d'indexes des dates, commence à 0.
    tpath = os.path.join(path, "tcompr")
    tfile      = io.Tfile(tpath)
    simulation["ldates"] = list(range(len(tfile.read())))


    # Déclaration de l'objet Pfile.
    ppath           = os.path.join(path, "pcompr")
    pfile = io.Pfile(ppath)


    # Lecture de la topologie du maillage : table de connectivité définissant les éléments
    conn = pfile.read_elements()

    # Lecture des indices des noeuds définissant les faces du domaine global
    facets = pfile.read_faces()

    # Extraction des indices des noeuds définissant la facette 1
    ids = np.where(facets[:,0]==1)[0]
    ids = np.unique(facets[ids,1:4].flatten())

    # Construction de la topologie inversée : dictionnaire qui à chaque noeud de la facette associe les indices globaux des éléments.
    vals = []                                             # liste des valeurs du dictionnaire
    for _ in range(len(ids)):                             # pour tout indice de noeud sur la facette 1...
        vals.append([])                                   # ... créer une liste vide

    rev = dict( zip(ids, vals) )            # créer le dictionnaire qui à chaque indice de noeud de la facette 1 associe la liste vide

    for i, el in enumerate(conn[:, 1:]):    # pour tout élément
        for p in el:                                      #     pour tout point dans l'élément
            if p in ids:                                  #         si le point est sur la facette
                rev[p].append(i)            #             ajouter l'élément à la liste associée au point


    # Lecture des coordonnées du maillage : ['x' 'z'] pour la facette 1 (c.f https://github.com/julienVLNT/drafts/blob/main/02_Pfile.ipynb)
    coords = pfile.read_coords(simulation["ldates"], ['x', 'z'])
    
    # Sélection des coordonnées des noeuds de la facette 1 uniquement
    coords = coords[:, :, ids-1]


    # Lecture des valeurs scalaires des champs.
    fields = pfile.read_fields(simulation["ldates"], pfile.fields)

    # Distribution des champs scalaires par tenseur : S puis E puis D.
    # fields[i, j, k] est la valeur au temps t_i du champs nom_j sur l'élément d'indice global el_k
    S = fields[:,   :6,  :]
    E = fields[:,  6:12, :]
    D = fields[:, 12:18, :]

    # Distribution de Peierls (même convention de sélection)
    P = fields[:,    -1, :]
    
    """
    # Pour chaque champ tensoriel "S", "E" puis "D"...
    for name in ["S", "E", "D"]:
        # Calculer la trace normalisée par la dimension `p`
        p   = simulation[name][:, :3, :].sum(axis=1) /3.0
        # Calculer le champ invariant J_2
        simulation[f"J2{name}"] = np.sqrt(   3./2 *np.power(simulation[name][:, :3, :] -1./3*p[:, np.newaxis, :], 2).sum(axis=1) \
                                           + 3. *np.power(simulation[name][:, 3:, :], 2).sum(axis=1)                             )
    """
    p   = S[:, :3, :].sum(axis=1) /3.0
    J2S = np.sqrt(   3./2 *np.power(  S[:, :3, :] -1./3*p[:, np.newaxis, :], 2).sum(axis=1) \
                                    + 3. *np.power(S[:, 3:, :], 2).sum(axis=1)              )
    
    p   = D[:, :3, :].sum(axis=1) /3.0
    J2D = np.sqrt(   3./2 *np.power(  D[:, :3, :] -1./3*p[:, np.newaxis, :], 2).sum(axis=1) \
                                    + 3. *np.power(D[:, 3:, :], 2).sum(axis=1)              )
    
    p   = E[:, :3, :].sum(axis=1) /3.0
    J2E = np.sqrt(   3./2 *np.power(  E[:, :3, :] -1./3*p[:, np.newaxis, :], 2).sum(axis=1) \
                                    + 3. *np.power(E[:, 3:, :], 2).sum(axis=1)              )

    # Calcul du travail
    W = (S * E).sum(axis=1)

    # Calcul des statistiques de chaque champ par date
    simulation["J2Smin"]  = np.nanmin(J2S, axis=1)
    simulation["J2Smax"]  = np.nanmax(J2S, axis=1)
    simulation["J2Smean"] = np.nanmean(J2S, axis=1)
    simulation["J2Sstd"]  = np.nanstd(J2S, axis=1)
    simulation["J2Sskew"] = sta.skew(J2S, axis=1, nan_policy='omit')
    simulation["J2Skurt"] = sta.kurtosis(J2S, axis=1, fisher=True, nan_policy='omit')

    simulation["J2Emin"]  = np.nanmin(J2E, axis=1)
    simulation["J2Emax"]  = np.nanmax(J2E, axis=1)
    simulation["J2Emean"] = np.nanmean(J2E, axis=1)
    simulation["J2Estd"]  = np.nanstd(J2E, axis=1)
    simulation["J2Eskew"] = sta.skew(J2E, axis=1, nan_policy='omit')
    simulation["J2Ekurt"] = sta.kurtosis(J2E, axis=1, fisher=True, nan_policy='omit')

    simulation["J2Dmin"]  = np.nanmin(J2D, axis=1)
    simulation["J2Dmax"]  = np.nanmax(J2D, axis=1)
    simulation["J2Dmean"] = np.nanmean(J2D, axis=1)
    simulation["J2Dstd"]  = np.nanstd(J2D, axis=1)
    simulation["J2Dskew"] = sta.skew(J2D, axis=1, nan_policy='omit')
    simulation["J2Dkurt"] = sta.kurtosis(J2D, axis=1, fisher=True, nan_policy='omit')

    simulation["Wmin"]  = np.nanmin(W, axis=1)
    simulation["Wmax"]  = np.nanmax(W, axis=1)
    simulation["Wmean"] = np.nanmean(W, axis=1)
    simulation["Wstd"]  = np.nanstd(W, axis=1)
    simulation["Wskew"] = sta.skew(W, axis=1, nan_policy='omit')
    simulation["Wkurt"] = sta.kurtosis(W, axis=1, fisher=True, nan_policy='omit')

    simulation["Pmin"]  = np.nanmin(P, axis=1)
    simulation["Pmax"]  = np.nanmax(P, axis=1)
    simulation["Pmean"] = np.nanmean(P, axis=1)
    simulation["Pstd"]  = np.nanstd(P, axis=1)
    simulation["Pskew"] = sta.skew(P, axis=1, nan_policy='omit')
    simulation["Pkurt"] = sta.kurtosis(P, axis=1, fisher=True, nan_policy='omit')


    # Passage à la représentation nodale de chaque champ élémentaire
    J2S = np.asarray([J2S[:, rev[id]].sum(axis=1)/len(rev[id]) for id in ids], dtype=np.float64)
    J2E = np.asarray([J2E[:, rev[id]].sum(axis=1)/len(rev[id]) for id in ids], dtype=np.float64)
    J2D = np.asarray([J2D[:, rev[id]].sum(axis=1)/len(rev[id]) for id in ids], dtype=np.float64)
    W   = np.asarray([W[:, rev[id]].sum(axis=1)/len(rev[id]) for id in ids], dtype=np.float64)
    P   = np.asarray([P[:, rev[id]].sum(axis=1)/len(rev[id]) for id in ids], dtype=np.float64)

    # Pour chaque pas de temps...
    for nt in simulation["ldates"]:

        # Construire la grille régulière correspondant au maillage au temps nt
        xz = coords[nt, :, :]
        xlin = np.linspace(xz[0,:].min(), xz[0,:].max(), 1024, endpoint=True)
        zlin = np.linspace(xz[1,:].min(), xz[1,:].max(), 1024, endpoint=True)
        xgrid, zgrid = np.meshgrid(xlin, zlin)

        # Construire l'interpolation des champs scalaires sur la grille régulière
        iJ2S = int.griddata( np.swapaxes(xz[[0,1], :], axis1=1, axis2=0), J2S.T[nt, :], (xgrid, zgrid), method='cubic' )
        iJ2E = int.griddata( np.swapaxes(xz[[0,1], :], axis1=1, axis2=0), J2E.T[nt, :], (xgrid, zgrid), method='cubic' )
        iJ2D = int.griddata( np.swapaxes(xz[[0,1], :], axis1=1, axis2=0), J2D.T[nt, :], (xgrid, zgrid), method='cubic' )
        iW   = int.griddata( np.swapaxes(xz[[0,1], :], axis1=1, axis2=0), W.T[nt, :],   (xgrid, zgrid), method='cubic' )
        iP   = int.griddata( np.swapaxes(xz[[0,1], :], axis1=1, axis2=0), P.T[nt, :],   (xgrid, zgrid), method='cubic' )


        # Traitement des NaNs
        iJ2S = np.nan_to_num(iJ2S, nan=np.nanmean(iJ2S))
        iJ2E = np.nan_to_num(iJ2E, nan=np.nanmean(iJ2E))
        iJ2D = np.nan_to_num(iJ2D, nan=np.nanmean(iJ2D))
        iW = np.nan_to_num(iW, nan=np.nanmean(iW))
        iP = np.nan_to_num(iP, nan=np.nanmean(iP))

        # Construire et exporter les images
        os.makedirs(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "J2S"), exist_ok=True)
        fig = plt.figure()
        img = fig.add_subplot(1,1,1)
        img.set_title(f"J2S(t{nt})")
        img.set_xlabel("X [px]")
        img.set_ylabel("Y [px]")
        img.imshow(iJ2S, origin="lower")
        fig.savefig(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "J2S", "J2S_"+str(nt).zfill(3)+".jpg"))
        plt.close()

        os.makedirs(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "J2E"), exist_ok=True)
        fig = plt.figure()
        img = fig.add_subplot(1,1,1)
        img.set_title(f"J2E(t{nt})")
        img.set_xlabel("X [px]")
        img.set_ylabel("Y [px]")
        img.imshow(iJ2E, origin="lower")
        fig.savefig(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "J2E", "J2E_"+str(nt).zfill(3)+".jpg"))
        plt.close()

        os.makedirs(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "J2D"), exist_ok=True)
        fig = plt.figure()
        img = fig.add_subplot(1,1,1)
        img.set_title(f"J2D(t{nt})")
        img.set_xlabel("X [px]")
        img.set_ylabel("Y [px]")
        img.imshow(iJ2D, origin="lower")
        fig.savefig(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "J2D", "J2D_"+str(nt).zfill(3)+".jpg"))
        plt.close()

        os.makedirs(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "Peierls"), exist_ok=True)
        fig = plt.figure()
        img = fig.add_subplot(1,1,1)
        img.set_title(f"Peierls(t{nt})")
        img.set_xlabel("X [px]")
        img.set_ylabel("Y [px]")
        img.imshow(iP, origin="lower")
        fig.savefig(os.path.join("analyse", "uniques", os.path.basename(path), "champs", f"Peierls", f"Peierls_"+str(nt).zfill(3)+".jpg"))
        plt.close()

        os.makedirs(os.path.join("analyse", "uniques", os.path.basename(path), "champs", "Work"), exist_ok=True)
        fig = plt.figure()
        img = fig.add_subplot(1,1,1)
        img.set_title(f"Work(t{nt})")
        img.set_xlabel("X [px]")
        img.set_ylabel("Y [px]")
        img.imshow(iW, origin="lower")
        fig.savefig(os.path.join("analyse", "uniques", os.path.basename(path), "champs", f"Work", f"Work_"+str(nt).zfill(3)+".jpg"))
        plt.close()



        # Analyse individuelle de la simulation via le champs de Peierls
        nrows = 1
        ncols = 5
        fig = plt.figure(figsize=(ncols*10+1, nrows*10))
        

        auto = pro.autocorrelation(iP)
        cinf = pro.cinfinity(iP)

        axe = fig.add_subplot(nrows, ncols, 1)
        axe.set_title("Peierls - $\mathcal{P}$")
        axe.set_xlabel("X [px]")
        axe.set_ylabel("Y [px]")
        axe.imshow(iP)

        axe = fig.add_subplot(nrows, ncols, 2)
        axe.set_title("$\mathcal{A}(\mathcal{P})$")
        axe.set_xlabel("X [px]")
        axe.set_ylabel("Y [px]")
        axe.imshow(auto)
        axe.contour(auto, [cinf, 1.0])

        
        angles = np.linspace(0, 179, 180)
        radius = np.zeros_like(angles) * np.nan
        for k in range(len(angles)):
            radius[k] = pro.length_vs_angle(auto, cinf, angles[k]*np.pi/180)

        axe = fig.add_subplot(nrows, ncols, 3)
        axe.set_title("Distance au contour $C_\infty$")
        axe.set_xlabel("$\\theta$ [°]")
        axe.set_ylabel("$R(\\theta)$ [px]")
        axe.plot(angles, radius)


        angle = angles[radius.argmax()]*np.pi/180 + np.pi/2
        r, a  = pro.radial_profile(auto, angle)
        
        axe = fig.add_subplot(nrows, ncols, 4)
        axe.set_title("Profil : $\\theta = $"+"{:.2f}".format(angle*180/np.pi)+"°")
        axe.set_xlabel("radius [px]")
        axe.set_ylabel("$\\mathcal{A}(\\mathcal{P})$ [px]")
        axe.plot(r, a)
        axe.hlines(y=cinf, xmin=0, xmax=r.max())


        angles_ = np.linspace(0, 2*np.pi, 360)
        radius_ = np.hstack((radius, radius))
        
        axe = fig.add_subplot(1, ncols, 5, projection="polar")
        axe.fill(angles_, radius_, color="tan")
        axe.set_rmax(radius.max())


        fig.savefig(os.path.join("analyse", "uniques", os.path.basename(path), "analyse_"+str(nt).zfill(3)+".jpg"))
        plt.close()

    
    # Ajouter le dictionnaire au dictionnaire du groupe.
    simulations["sim"].append(simulation)
    print()


# TRAITEMENT DES STATISTIQUES DU GROUPE
print("> Statistiques de groupe")
print("> ----------------------")
print()

# colors = ["black", "blue", "gray", "green", "olive", "orange", "red", "yellow"]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#0cff0c', '#f7879a', '#047495', '#fec615', "#cba560"]
for sim in simulations["sim"]:
    sim["color"] = colors.pop( np.random.randint(0, len(colors)) )

nrows = 7    # Nombre d'indicateurs statistiques plus 1
ncols = 5
fig = plt.figure(figsize=(ncols*(10+1), nrows*(10+1)))


axe = fig.add_subplot(nrows, ncols, 1)
axe.set_title("$J_2(\\sigma)$", fontsize=40)
axe.set_ylabel("MIN", fontsize=35)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Smin"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 2)
axe.set_title("$J_2(\\epsilon)$", fontsize=40)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Emin"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 3)
axe.set_title("$J_2(\\dot\\epsilon)$", fontsize=40)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Dmin"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 4)
axe.set_title("$\\mathcal{P}$", fontsize=40)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Pmin"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 5)
axe.set_title("$\\mathcal{W}$", fontsize=40)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Wmin"], color=sim["color"])


axe = fig.add_subplot(nrows, ncols, 6)
axe.set_ylabel("MAX", fontsize=35)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Smax"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 7)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Emax"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 8)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Dmax"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 9)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Pmax"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 10)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Wmax"], color=sim["color"])


axe = fig.add_subplot(nrows, ncols, 11)
axe.set_ylabel("MEAN", fontsize=35)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Smean"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 12)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Emean"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 13)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Dmean"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 14)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Pmean"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 15)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Wmean"], color=sim["color"])


axe = fig.add_subplot(nrows, ncols, 16)
axe.set_ylabel("STD", fontsize=35)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Sstd"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 17)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Estd"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 18)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Dstd"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 19)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Pstd"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 20)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Wstd"], color=sim["color"])


axe = fig.add_subplot(nrows, ncols, 21)
axe.set_ylabel("SKEW", fontsize=35)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Sskew"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 22)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Eskew"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 23)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Dskew"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 24)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Pskew"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 25)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Wskew"], color=sim["color"])


axe = fig.add_subplot(nrows, ncols, 26)
axe.set_xlabel("Temps [index]")
axe.set_ylabel("KURTOSIS", fontsize=35)
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Skurt"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 27)
axe.set_xlabel("Temps [index]")
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Ekurt"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 28)
axe.set_xlabel("Temps [index]")
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["J2Dkurt"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 29)
axe.set_xlabel("Temps [index]")
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Pkurt"], color=sim["color"])

axe = fig.add_subplot(nrows, ncols, 30)
axe.set_xlabel("Temps [index]")
for sim in simulations["sim"]:
    axe.plot(sim["ldates"], sim["Wkurt"], color=sim["color"])


axe = fig.add_subplot(nrows, ncols, 31)
axe.set_axis_off()
for k, sim in enumerate(simulations["sim"]):
    axe.text(x=0, y=0.95 - 0.1*k, s=sim["name"], color=sim["color"], wrap=True, fontsize=50)


fig.savefig(os.path.join("analyse", "groupes", f"{simulations['name']}_analyse.jpg"))
plt.close()

print(" ----- ")
print("| FIN |")
print(" ----- ")
