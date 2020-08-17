from lxml import etree

#[i.tag for i in res[0]] #iterate children
#[i.tag for i in res[0].iter(<tag>)] #iterate recursively
#[i.attrib for i in res[0].iter("assignment")]
# loop -> header #iterate var, upper/lower bound
# assignment -> [target, value]
# target -> name -> subscripts
# read lists: res[0].iter("name")
# temp var lists: only read on write in the same loop
# need to exclude subscript and assignment target from read lists
# selector.getpath(node)


def parse_xml():
    root = etree.parse("fortran/samfshalconv.xml")
    return root


def visit_loop(node):
    indexvars = node.xpath(".//index-variable")
    indexvar_ids = [i.get("name") for i in indexvars]
    writevars = node.xpath(".//assignment/target/name")
    readvars = [i for i in node.xpath(".//*[not(name()='target')]/name")
                if i.get("id") not in indexvar_ids]
    
    return indexvars, writevars, readvars


def find_vars_in_range(node, startlineno, endlineno):
    writevars = [i for i in node.xpath(".//assignment/target/name")
                 if int(i.get("line_begin")) >= startlineno and int(i.get("line_begin")) <= endlineno]
    readvars  = [i for i in node.xpath(".//*[not(name()='target')]/name")
                 if int(i.get("line_begin")) >= startlineno and int(i.get("line_begin")) <= endlineno]
    
    return writevars, readvars


def get_declared_vars(root = parse_xml()):
    return [i.get("name") for i in root.xpath(".//variables/variable")]


def get_declared_arr_vars(root = parse_xml()):
    return [i.get("name") for i in root.xpath(".//variables/variable[dimensions]")]


def get_nonfloat_arr_vars(root = parse_xml()):
    return [i.get("name") for i in
            root.xpath(".//declaration[type[@name='integer' or @name='logical']]/variables/variable[dimensions]")]


def get_arguments_range(startlineno, endlineno):
    root = parse_xml()
    declaredarrvars = get_declared_arr_vars(root)
    writevars, readvars = find_vars_in_range(root, startlineno, endlineno)
    writevars = set([i.get("id") for i in writevars])
    readvars = set([i.get("id") for i in readvars
                    if i.get("id") not in ["n","i","k","min","max","fpvs","fpvsx"]])
    readvars_list = [i for i in declaredarrvars if i in readvars]
    writevars_list = [i for i in declaredarrvars if i in writevars]
    
    return writevars_list, readvars_list, set(readvars_list).intersection(writevars_list)


def print_ser_def(varlist):
    return " ".join([var+"="+var for var in varlist])


#root = etree.parse("fortran/samfshalconv.xml")
#res = root.xpath('/ofp/file/subroutine/body/loop')
part1_range = (218, 458)
part2_range = (459, 1300)
part3_range = (1301, 1513)
part4_range = (1517, 1810)
