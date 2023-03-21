import numpy as np



class Pfile():

    def __init__(self, path: str) -> None:
        self.path = path
        self.__read_header()
        
        return None
    

    def __read_header(self) -> None:

        fields   = []
        topology = []

        try:
            with open(self.path, "r") as stream:
                
                # Read fields related informations
                number = int(stream.readline().strip())
                for _ in range(number):
                    fields.append(stream.readline().strip())

                # Read topology related informations
                next(stream)
                topology = np.int_(list(filter(None, stream.readline().strip().split(" "))))

                # Update object attributes
                self.nfields = len(fields)
                self.fields  = fields

                self.nelem = topology[0]
                self.nvert = topology[1]
                self.ngaus = topology[2]
                self.ndime = topology[3]
                self.nface = topology[4]
                self.npres = topology[5]

                self.offset = 1 +self.nfields +3 +self.nelem +1 +(self.npres//10 +1) +1 +self.nface +2

        except FileNotFoundError:
            print(f"{self.path} is not readable, skip.")
        
        return None
    

    def read_elements(self) -> np.ndarray:
        elements = None

        try:
            elements = np.loadtxt( self.path,
                                   dtype=int,
                                   skiprows=(4 +self.nfields),
                                   max_rows=self.nelem )
            elements = elements[:, 1:]
        
        except FileNotFoundError:
            print(f"{self.path} is not readable, skip.")

        return elements
    

    def read_contour(self) -> np.ndarray:
        # contour = None

        # try:
        #     contour = np.genfromtxt( self.path,
        #                               dtype=int,
        #                               skip_header=(4 +self.nfields +self.nelem +1),
        #                               skip_footer=(self.npres//10 +1) )
        #     contour = contour.flatten()

        # except FileNotFoundError:
        #     print(f"{self.path} is not readable, skip.")

        # return contour
        raise NotImplementedError


    def read_faces(self) -> np.ndarray:
        faces = None

        try:
            faces = np.loadtxt( self.path,
                                dtype=int,
                                skiprows=(4 +self.nfields +self.nelem +1 +int(self.npres//10 +1) +1),
                                max_rows=self.nface )
            faces = faces[:, 1:]

        except FileNotFoundError:
            print(f"{self.path} is not readable, skip.")

        return faces
    

    def read_coords(self, dates: list, names: list) -> np.ndarray:
        coords = np.zeros((len(dates), len(names), self.nvert)) *float('nan')

        coordsmap = { 'x' : 1, 'y' : 2, 'z' : 3,
                      'vx': 4, 'vy': 5, 'vz': 6,
                      'ux': 7, 'uy': 8, 'uz': 9,
                      'T' : 10 }
        
        try:
            for i, n in enumerate(dates):
                coords[i, :, :] = np.loadtxt( self.path, 
                                              dtype=float,
                                              skiprows=(self.offset +n*(self.nvert + self.nelem +3)),
                                              max_rows=self.nvert,
                                              usecols=[coordsmap[name] for name in names] ).transpose()
        
        except FileNotFoundError:
            print(f"{self.path} is not readable, skip.")

        return coords
    

    def read_fields(self, dates: list, names: list) -> np.ndarray:
        fields = np.zeros((len(dates), len(names), self.nelem)) *float('nan')

        fieldsmap = dict(
            zip(
                    [name for name in self.fields],
                    [self.fields.index(name)+1 for name in self.fields] 
               )
        )

        try:
            for i, n in enumerate(dates):
                fields[i, :, :] = np.loadtxt( self.path,
                                              dtype=float,
                                              skiprows=(self.offset +n*(self.nelem +3) +(n+1)*self.nvert),
                                              max_rows=self.nelem,
                                              usecols=[fieldsmap[name] for name in names] ).transpose()
                
        except FileNotFoundError:
            print(f"{self.path} is not readable, skip.")

        return fields
    


class Tfile():

    def __init__(self, path: str) -> None:
        self.path = path

        return None


    def read(self) -> np.ndarray:
        dates = None

        try:
            dates = np.genfromtxt(self.path)
            dates = dates[:, 1]

        except FileNotFoundError:
            print(f"{self.path} is not readable, skip.")

        return dates
