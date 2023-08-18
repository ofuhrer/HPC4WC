from dsl.frontend.language import Horizontal, Vertical
from dsl.frontend.parser import parse_function


def example_function():

    with Iterations[1:num_iter-1]:

        #Halo Update:
        # bottom edge (without corners)
        in_field[:, :num_halo, num_halo:-num_halo] = in_field[
                                                  :, -2 * num_halo: -num_halo, num_halo:-num_halo
                                                  ]

        # top edge (without corners)
        in_field[:, -num_halo:, num_halo:-num_halo] = in_field[
                                                   :, num_halo: 2 * num_halo, num_halo:-num_halo
                                                   ]

        # left edge (including corners)
        in_field[:, :, :num_halo] = in_field[:, :, -2 * num_halo: -num_halo]

        # right edge (including corners)
        in_field[:, :, -num_halo:] = in_field[:, :, num_halo: 2 * num_halo]


        #First Laplacian:
        with Vertical[1:nz]: #überall ein +1 dazu weil laplacian code ist inspired von fortran version und da ist do1,10 incl 10
            with Horizontal[num_halo : ny+num_halo+1, num_halo : nx+num_halo+1]:
                tmp_field[i,j,k] = lap(in_field)

        #Second Laplacian:
        with Vertical[1:nz]:
            with Horizontal[1+num_halo : ny+num_halo, 1+num_halo : nx+num_halo]:
                out_field[i,j,k] = lap(tmp_field)

        #Achtung ab hier sollte es nicht mehr im body vom with vertical with horizontal sein!!

        #Updating Out Field I guess
        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
                in_field[:, num_halo:-num_halo, num_halo:-num_halo]
                - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )


        #in_field, out_field = out_field, in_field: Hier mit tmp field gelöst (das wird nachher eh neu gfüllt)
        tmp_field = in_field
        in_field = out_field
        out_field = tmp_field



    #LAST "ITERATION"
    #Halo Update:
    # bottom edge (without corners)
    in_field[:, :num_halo, num_halo:-num_halo] = in_field[
                                              :, -2 * num_halo: -num_halo, num_halo:-num_halo
                                              ]

    # top edge (without corners)
    in_field[:, -num_halo:, num_halo:-num_halo] = in_field[
                                               :, num_halo: 2 * num_halo, num_halo:-num_halo
                                               ]

    # left edge (including corners)
    in_field[:, :, :num_halo] = in_field[:, :, -2 * num_halo: -num_halo]

    # right edge (including corners)
    in_field[:, :, -num_halo:] = in_field[:, :, num_halo: 2 * num_halo]


    #First Laplacian:
    with Vertical[1:nz]:
        with Horizontal[num_halo : ny+num_halo+1, num_halo : nx+num_halo+1]:
            tmp_field[i,j,k] = lap(in_field)

    #Second Laplacian:
    with Vertical[1:nz]:
        with Horizontal[1+num_halo : ny+num_halo, 1+num_halo : nx+num_halo]:
            out_field[i,j,k] = lap(tmp_field)

    #Achtung ab hier sollte es nicht mehr im body vom with vertical with horizontal sein!!

    #Updating Out Field I guess
    out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
    )

    #ANOTHER FINAL HALO UPDATE:
    # Halo Update:
    # bottom edge (without corners)
    in_field[:, :num_halo, num_halo:-num_halo] = in_field[
                                                 :, -2 * num_halo: -num_halo, num_halo:-num_halo
                                                 ]

    # top edge (without corners)
    in_field[:, -num_halo:, num_halo:-num_halo] = in_field[
                                                  :, num_halo: 2 * num_halo, num_halo:-num_halo
                                                  ]

    # left edge (including corners)
    in_field[:, :, :num_halo] = in_field[:, :, -2 * num_halo: -num_halo]

    # right edge (including corners)
    in_field[:, :, -num_halo:] = in_field[:, :, num_halo: 2 * num_halo]



if __name__ == "__main__":
    parse_function(example_function)


