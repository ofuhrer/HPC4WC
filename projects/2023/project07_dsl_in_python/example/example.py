from dsl.frontend.language import Horizontal, Vertical
from dsl.frontend.parser import parse_function


def example_function():
    with Vertical[1:10]:
        with Horizontal[1:10,1:10]:
            field=lap(field)

    #Jetzt chemmer s halo update entweder direkt wie im stencil mache:
    field[:, :num_halo, num_halo:-num_halo] = field[
                                              :, -2 * num_halo: -num_halo, num_halo:-num_halo
                                              ] #to be fair mir brüchtet da no multiplikation wegemm -2*num_halo

    #Oder eso:
    with Vertical[num_halo:(nz - num_halo)]:
        with Horizontal[0:nx, 0:num_halo]:
            field = update_halo_bottom_edge(field)

    #Ich find die erst Version eleganter. PS: I beidne Fäll isch das jz mal nur für bottom edge.


if __name__ == "__main__":
    parse_function(example_function)
