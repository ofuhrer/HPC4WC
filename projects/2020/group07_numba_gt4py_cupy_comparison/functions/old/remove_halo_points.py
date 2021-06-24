#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:11:05 2020

@author: lau
"""



def remove_halo_points(dim, field, num_halo):
    """Removes halo points to an array on each end (call only once after timeloop before save)"""
    
    if dim == 1:
        out_field = field[num_halo:-num_halo] 
    
    if dim == 2:
        out_field = field[num_halo:-num_halo,num_halo:-num_halo]
    
    if dim == 3:
        out_field = field[num_halo:-num_halo,num_halo:-num_halo,num_halo:-num_halo]
    
    return out_field