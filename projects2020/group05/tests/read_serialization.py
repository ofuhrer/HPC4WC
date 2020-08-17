from shalconv import DATAPATH
from shalconv.serialization import data_dict_from_var_list, numpy_dict_to_gt4py_dict
import serialbox as ser


def read_serialization_partx(var_list, part, tile = 0, path = DATAPATH):
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Serialized_rank"+str(tile))
    sp         = ser.Savepoint(f"samfshalcnv-part{part}-input")
    data       = data_dict_from_var_list(var_list, serializer, sp)
    
    return numpy_dict_to_gt4py_dict(data)


def read_serialization_part2_x(var_list, part, tile = 0, path = DATAPATH):
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Serialized_rank"+str(tile))
    sp         = ser.Savepoint(f"samfshalcnv-part2-{part}")
    data       = data_dict_from_var_list(var_list, serializer, sp)
    
    return numpy_dict_to_gt4py_dict(data)


def read_serialization_part2():
    var_list = ['ix', 'km', 'islimsk', 'dot', 'qtr', 'kpbl', 'kb', 'kbcon', 'kbcon1',
                'cnvwt', 'dellal', 'ktconn', 'pwo', 'qlko_ktcon', 'qrcko', 'xmbmax', #output
                'ktcon', 'ktcon1', 'kbm', 'kmax', 'aa1', 'cina', 'tkemean',
                'clamt', 'del', 'edt', 'pdot', 'po', 'hmax', 'vshear', 'xlamud',
                'pfld', 'to', 'qo', 'uo', 'vo', 'qeso', 'ctro', 'wu2', 'buo',
                'drag', 'wc', 'dbyo', 'zo', 'xlamue', 'heo', 'heso', 'hcko',
                'ucko', 'vcko', 'qcko', 'ecko', 'eta', 'zi', 'c0t', 'sumx',
                'cnvflg', 'flg']
                
    return read_serialization_partx(var_list, 2)


def read_serialization_part2_1():
    var_list = ['ix','km','hmax', 'heo', 'heso', 'kb', 'to', 'qeso', 'po', 'qo', 'uo', 'vo']
    
    return read_serialization_part2_x(var_list, 1)


def read_serialization_part2_2():
    var_list = ['ix','km','cnvflg','pdot']
    
    return read_serialization_part2_x(var_list, 2)


def read_serialization_part2_3():
    var_list = ['ix','km','clamt','xlamud','xlamue','eta','kmax','kbm','hcko','ucko','vcko','tkemean','sumx',
                'cnvflg','kb','zi','heo','dbyo','heso','pgcon','uo','vo']
    
    return read_serialization_part2_x(var_list, 3)


def read_serialization_part2_4():
    var_list = ['ix','km','hcko','dbyo','ucko','vcko',
                'cnvflg','kmax','kbm','kbcon','kbcon1','flg','pfld']
    
    return read_serialization_part2_x(var_list, 4)


def read_serialization_part2_5():
    var_list = ['ix','km','cnvflg','kbcon1','flg',
                'cina','kb','zo','qeso','to','dbyo','qo','pdot','islimsk']
    
    return read_serialization_part2_x(var_list, 5)


def read_serialization_part2_6():
    var_list = ['ix','km','cnvflg','cina']
    
    return read_serialization_part2_x(var_list, 6)


def read_serialization_part3():
    var_list = ['ix', 'km', 'delp', 'garea', 'qtr', 'u1', 'v1', 'kb',
                'kbcon', 'kbcon1', 'ktcon', 'ktcon1', 'kmax', 'del',
                'umean', 'tauadv', 'gdx', 'dtconv', 'po', 'xlamud',
                'xmb', 'xmbmax', 'to', 'qo', 'uo', 'vo', 'ctro', #'qaero',
                'wc', 'scaldfunc', 'sigmagfm', 'qlko_ktcon', 'xlamue',
                'heo', 'dellah', 'dellaq', 'dellae', 'dellau', 'dellav', 'dellal',
                'hcko', 'ucko', 'vcko', 'qcko', 'qrcko', 'ecko', 'eta',
                'zi', 'c0t', 'sumx', 'cnvflg']
    
    return read_serialization_partx(var_list, 3)


def read_serialization_part4():
    var_list = ['ix', 'km', 'islimsk', 'qtr', 'q1', 't1', 'u1', 'v1',
                'cnvw', 'dt_mf', 'kbot', 'kcnv', #output
                'ktop', 'rn', 'cnvc', 'ud_mf', 'kb', 'kbcon', 'ktcon',
                'kmax', 'del', 'delhbar', 'delq', 'delq2', 'delqbar',
                'delqev', 'deltbar', 'deltv', 'edt', 'qcond', 'qevap',
                'rntot', 'xmb', 'delubar', 'delvbar', 'pfld', #delebar(ix,ntr)
                'qeso', 'ctr', 'sigmagfm', 'dellal', 'dellah', #'qaero',
                'dellaq', 'dellae', 'dellau', 'dellav', 'eta', 'pwo',
                'cnvwt', 'cnvflg', 'flg']
    
    return read_serialization_partx(var_list, 4)


if __name__ == "__main__":
    read_serialization_part2()
    read_serialization_part3()
    read_serialization_part4()
