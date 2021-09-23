"New functions created by us. Only for illustration, cannot be executed like this in this script. They are copied out of the partitioner.py line 624 to 779"

def lr_boundary_seq(self, rank: int):
    """Calculates the sequence of boundaries on an edge of a tile + the amount of to_ranks shared with this rank"""
    rank_position_y = (rank % self.tile.total_ranks) // self.tile.layout[0] # position of rank on left/right boundary in y-axis
    boundary_seq = []
    if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0: # layout is dividable
        if self.tile.layout[0] >= self.tile.layout[1]:
            div = self.tile.layout[0] // self.tile.layout[1]
        else:
            div = 1
        boundary_seq = [div]*self.tile.layout[1]   # adds sequence of repeated multiples (div) in a list.
        Nboundary = boundary_seq[rank_position_y]
    else:
        nfirst = self.tile.layout[0] // self.tile.layout[1] + 1
        boundary_seq.append(nfirst)
        for y in range(1,self.tile.layout[1]): # calculates the rest of the sequence of the boundary by dividing the layout
            num1 = (self.tile.layout[0]*self.tile.layout[1]) - y*self.tile.layout[0]
            num2before = num1//self.tile.layout[1] + 1
            num2after = (num1 - self.tile.layout[0])//self.tile.layout[1] + 1
            numadd = num2before - num2after + 1
            boundary_seq.append(numadd)
        Nboundary = boundary_seq[rank_position_y] # Amount of boundaries created on the edge of a rank
    return Nboundary, boundary_seq

def ul_boundary_seq(self, rank: int): 
    """Calculates the sequence of boundaries on an edge of a tile + the amount of to_ranks shared with this rank"""
    rank_position_x = (rank % self.tile.total_ranks)//self.tile.layout[0] # position of rank on upper/ower boundary in x-axis
    boundary_seq = []
    if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0: # layout is dividable
        if self.tile.layout[1] >= self.tile.layout[0]:
            div = self.tile.layout[1]//self.tile.layout[0]
        else:
            div = 1
        boundary_seq = [div]*self.tile.layout[0]
        Nboundary = boundary_seq[rank_position_x]
    else:
        nfirst = self.tile.layout[1] // self.tile.layout[0] + 1
        boundary_seq.append(nfirst)
        for x in range(1,self.tile.layout[0]): # calculates the rest of the sequence of the boundary by dividing the layout
            num1 = (self.tile.layout[0]*self.tile.layout[1]) - x*self.tile.layout[1]  
            num2before = num1//self.tile.layout[0] + 1      
            num2after = (num1-self.tile.layout[1])//self.tile.layout[0] + 1   
            numadd = num2before - num2after + 1  
            boundary_seq.append(numadd)
        Nboundary = boundary_seq[rank_position_x] # Amount of boundaries created on the edge of a rank
    return Nboundary, boundary_seq

def lr_to_ranks(self, rank: int, to_root_rank: int):
    """Returns the to_ranks of the boundaries for a specific rank and its edge"""
    Nboundary, boundary_seq =  self.lr_boundary_seq(rank)
    if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0:
        div = self.tile.layout[1]//self.tile.layout[0]
        boundary_seq = boundary_seq
        if(div >= 1):
            cumsum_boundary_seq = np.cumsum(boundary_seq)//div # cumulated sum of boundary sequence to calculate shared ranks later on
        else:
            cumsum_boundary_seq = np.cumsum(boundary_seq)
    else:
        boundary_seq = [x - 1 for x in boundary_seq]
        cumsum_boundary_seq = np.cumsum(boundary_seq) # a cumulative sum of boundary_seq to know the start and end of the shared ranks 
    rank_position_y = (rank % self.tile.total_ranks) // self.tile.layout[0]
    to_root_rank = to_root_rank
    to_ranks_pot = []
    for x in range(self.tile.layout[0]):
        if self.tile.on_tile_left(rank):
            to_ranks_pot.append(to_root_rank + (self.tile.layout[0]*(self.tile.layout[1]-1)) + x)
        else:
            to_ranks_pot.append(to_root_rank + x) # creates list of all sharable ranks (potential to_ranks) on the respective edge
    to_ranks_pot = np.fliplr([to_ranks_pot, to_ranks_pot, to_ranks_pot])[1] # flips list for orientation purposes
    if(rank_position_y == 0):
        if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0:
            if self.tile.layout[0]//self.tile.layout[1] >= 1:
                start = 0
                end = cumsum_boundary_seq[0]
                to_ranks_seq = to_ranks_pot[start:end]
            else:
                start = 0
                div = self.tile.layout[1]//self.tile.layout[0]
                end = cumsum_boundary_seq[0]//div
                to_ranks_seq = to_ranks_pot[start:end] 
        else:
            start = 0
            end = cumsum_boundary_seq[0]+1
            to_ranks_seq = to_ranks_pot[start:end]
    else:
        if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
            if self.tile.layout[0]/self.tile.layout[1] >= 1:
                start = cumsum_boundary_seq[rank_position_y-1]
                end = cumsum_boundary_seq[rank_position_y]
                to_ranks_seq = to_ranks_pot[start:end]
            else:
                div = self.tile.layout[1]//self.tile.layout[0]
                start = rank_position_y//div
                end = start + 1
                to_ranks_seq = to_ranks_pot[start:end]
        else:
            start = cumsum_boundary_seq[rank_position_y-1]
            end = cumsum_boundary_seq[rank_position_y] + 1
            to_ranks_seq = to_ranks_pot[start:end]
    return to_ranks_seq

def ul_to_ranks(self, rank: int, to_root_rank: int):
    """Returns the to_ranks of the boundaries for a specific rank and its edge"""
    Nboundary, boundary_seq =  self.ul_boundary_seq(rank)
    if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
        div = self.tile.layout[1]//self.tile.layout[0]
        boundary_seq = boundary_seq
        if(div >= 1):
            cumsum_boundary_seq = np.cumsum(boundary_seq)//div # cumulated sum of boundary sequence to calculate shared ranks later on
        else:
            cumsum_boundary_seq = np.cumsum(boundary_seq)
    else:
        boundary_seq = [x - 1 for x in boundary_seq]
        cumsum_boundary_seq = np.cumsum(boundary_seq) # a cumulative sum of boundary_seq to know the start and end of the shared ranks 
    rank_position_x = (rank % self.tile.total_ranks)%self.tile.layout[0]
    to_root_rank = to_root_rank
    to_ranks_pot = []
    for y in range(self.tile.layout[1]):
        if self.tile.on_tile_top(rank):
            to_ranks_pot.append(to_root_rank + self.tile.layout[0]*y) # creates list of all sharable ranks (potential to_ranks) on the respective edge
        else:
            to_ranks_pot.append(to_root_rank + self.tile.layout[0]*y + self.tile.layout[0]-1)
                
    to_ranks_pot = np.fliplr([to_ranks_pot,to_ranks_pot,to_ranks_pot])[1] # flips list for orientation         
    if(rank_position_x == 0):
        if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
            if self.tile.layout[1]//self.tile.layout[0] > 1:
                div = self.tile.layout[1]//self.tile.layout[0]
                start = 0
                end = start + div
                to_ranks_seq = to_ranks_pot[start:end]
            else:
                div = self.tile.layout[0]//self.tile.layout[1]
                start = 0
                end = start + 1
                to_ranks_seq = to_ranks_pot[start:end]
        else:
            start = 0
            end = cumsum_boundary_seq[0]+1
            to_ranks_seq = to_ranks_pot[start:end]
    else:
        if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
            if self.tile.layout[1]//self.tile.layout[0] > 1:
                div = self.tile.layout[1]//self.tile.layout[0]
                start = rank_position_x*div
                end = start + div
                to_ranks_seq = to_ranks_pot[start:end]
            else:
                div = self.tile.layout[0]//self.tile.layout[1]
                start = rank_position_x//div
                end = start + 1
                to_ranks_seq = to_ranks_pot[start:end]
        else:
            start = cumsum_boundary_seq[rank_position_x-1]
            end = cumsum_boundary_seq[rank_position_x] + 1
            to_ranks_seq = to_ranks_pot[start:end]
    return to_ranks_seq