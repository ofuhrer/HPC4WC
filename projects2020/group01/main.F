program test
    use m_countdown, only : cdstart, cdshow, cdstop
    use machine, only : kind_phys
    use funcphys, only : gfuncphys
    use module_sfc_sice, only : sfc_sice
    implicit none

    integer :: im, km, ipr
    integer, dimension(:), allocatable :: islimsk, ones

    real (kind=kind_phys) :: delt, cimin
    real (kind=kind_phys), dimension(:), allocatable :: ps, t1, q1, &
        sfcemis, dlwflx, sfcnsw, sfcdsw, srflag, cm, ch, prsl1, prslki, &
        wind, hice, fice, tice, weasd, tskin, tprcp, ep, snwdph, &
        qsurf, snowmt, gflux, cmm, chh, evap, hflx, &
        hice_ref, fice_ref, tice_ref, weasd_ref, tskin_ref, tprcp_ref, &
        ep_ref, snwdph_ref, qsurf_ref, snowmt_ref, gflux_ref, cmm_ref, &
        chh_ref, evap_ref, hflx_ref
    real (kind=kind_phys), dimension(:, :), allocatable :: stc, stc_ref

    logical :: lprnt
    logical, dimension(:), allocatable :: flag_iter, flag

    integer :: iter, i
    integer :: tile, num_tiles
    integer :: ser_count, ser_count_max
    character(len=6) :: ser_count_str

    real(8) :: time1, time2

    ser_count_max = 19
    num_tiles = 6

    ! initialize
    call gfuncphys()

    call cdstart(num_tiles * ser_count_max * 2)

    do tile = 0, num_tiles - 1

    !$ser init directory='./dump' prefix='Serialized' directory_ref='./data' prefix_ref='Generator' mpi_rank=tile

    do ser_count = 0, ser_count_max
    do iter = 1, 2

        call cdshow(tile * ser_count_max + ser_count * 2 + iter - 1)
        
        write(ser_count_str, '(i6.6)') ser_count

        !$ser mode read
        !$ser verbatim if (iter == 1) then
        !$ser savepoint "sfc_sice-in-iter1-"//trim(ser_count_str)
        !$ser verbatim else
        !$ser savepoint "sfc_sice-in-iter2-"//trim(ser_count_str)
        !$ser verbatim end if
        !$ser data im=im km=km ipr=ipr lprnt=lprnt

        allocate(ps(im), t1(im), q1(im), sfcemis(im), dlwflx(im), sfcnsw(im), &
            sfcdsw(im), srflag(im), cm(im), ch(im), prsl1(im), prslki(im), &
            wind(im), hice(im), fice(im), tice(im), weasd(im), tskin(im), &
            tprcp(im), ep(im), snwdph(im), qsurf(im), snowmt(im), gflux(im), &
            cmm(im), chh(im), evap(im), hflx(im), stc(im, km), flag_iter(im), &
            islimsk(im), &
            hice_ref(im), fice_ref(im), tice_ref(im), weasd_ref(im), tskin_ref(im), &
            tprcp_ref(im), stc_ref(im, km), ep_ref(im), snwdph_ref(im), &
            qsurf_ref(im), snowmt_ref(im), gflux_ref(im), cmm_ref(im), &
            chh_ref(im), evap_ref(im), hflx_ref(im), &
            ones(im), flag(im))
        ones(:) = 1

        ! input
        !$ser data ps=ps t1=t1 q1=q1 delt=delt sfcemis=sfcemis dlwflx=dlwflx
        !$ser data sfcnsw=sfcnsw sfcdsw=sfcdsw srflag=srflag cm=cm ch=ch
        !$ser data prsl1=prsl1 prslki=prslki islimsk=islimsk wind=wind
        !$ser data flag_iter=flag_iter cimin=cimin
        ! inout
        !$ser data hice=hice fice=fice tice=tice weasd=weasd
        !$ser data tskin=tskin tprcp=tprcp stc=stc ep=ep
        !$ser data snwdph=snwdph qsurf=qsurf snowmt=snowmt
        !$ser data gflux=gflux cmm=cmm chh=chh evap=evap hflx=hflx

        !$ser mode write
        !$ser verbatim if (iter == 1) then
        !$ser savepoint "sfc_sice-inside-iter1-"//trim(ser_count_str)
        !$ser verbatim else
        !$ser savepoint "sfc_sice-inside-iter2-"//trim(ser_count_str)
        !$ser verbatim end if        

        do i = 1, im
            flag(i) = (islimsk(i) == 2) .and. flag_iter(i)
        enddo
        write(*,'(I5)', advance="no") sum(ones, flag)
        call tic(time1, time2)

        call sfc_sice                                                  &
            !  ---  inputs:
            ( im, km, ps, t1, q1, delt,                                &
            sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,                   &
            cm, ch, prsl1, prslki, islimsk, wind,                      &
            flag_iter, lprnt, ipr, cimin,                              &
            !  ---  input/outputs:
            hice, fice, tice, weasd, tskin, tprcp, stc, ep,            &
            snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx )

        call toc(time1, time2)

        !$ser mode read
        !$ser verbatim if (iter == 1) then
        !$ser savepoint "sfc_sice-out-iter1-"//trim(ser_count_str)
        !$ser verbatim else
        !$ser savepoint "sfc_sice-out-iter2-"//trim(ser_count_str)
        !$ser verbatim end if
        ! inout
        !$ser data hice=hice_ref fice=fice_ref tice=tice_ref weasd=weasd_ref
        !$ser data tskin=tskin_ref tprcp=tprcp_ref stc=stc_ref ep=ep_ref
        !$ser data snwdph=snwdph_ref qsurf=qsurf_ref snowmt=snowmt_ref
        !$ser data gflux=gflux_ref cmm=cmm_ref chh=chh_ref evap=evap_ref hflx=hflx_ref

        call check_r_1d(hice, hice_ref, "hice")
        call check_r_1d(fice, fice_ref, "fice")
        call check_r_1d(tice, tice_ref, "tice")
        call check_r_1d(weasd, weasd_ref, "weasd")
        call check_r_1d(tskin, tskin_ref, "tskin")
        call check_r_1d(tprcp, tprcp_ref, "tprcp")
        call check_r_2d(stc, stc_ref, "stc")
        call check_r_1d(ep, ep_ref, "ep")
        call check_r_1d(snwdph, snwdph_ref, "snwdph")
        call check_r_1d(qsurf, qsurf_ref, "qsurf")
        call check_r_1d(snowmt, snowmt_ref, "snowmt")
        call check_r_1d(gflux, gflux_ref, "gflux")
        call check_r_1d(cmm, cmm_ref, "cmm")
        call check_r_1d(chh, chh_ref, "chh")
        call check_r_1d(evap, evap_ref, "evap")
        call check_r_1d(hflx, hflx_ref, "hflx")

        deallocate(ps, t1, q1, sfcemis, dlwflx, sfcnsw, &
            sfcdsw, srflag, cm, ch, prsl1, prslki, &
            wind, hice, fice, tice, weasd, tskin, &
            tprcp, ep, snwdph, qsurf, snowmt, gflux, &
            cmm, chh, evap, hflx, stc, flag_iter, &
            islimsk, &
            hice_ref, fice_ref, tice_ref, weasd_ref, tskin_ref, &
            tprcp_ref, stc_ref, ep_ref, snwdph_ref, &
            qsurf_ref, snowmt_ref, gflux_ref, cmm_ref, &
            chh_ref, evap_ref, hflx_ref, ones, flag)

    end do
    end do

    !$ser cleanup

    end do

    call cdstop()

    write(*,*) 'FINISHED!'

contains

    subroutine check_r_2d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_r_2d

    subroutine check_r_1d(a, b, name, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a(:), b(:)
        character(len=*), intent(in) :: name
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        logical :: close

        close = all(isclose_r(a, b, atol, rtol))
        if (.not. close) then
            write(*,*) 'ERROR: ' // trim(name) // ' does not validate', tile, ser_count
        end if

    end subroutine check_r_1d

    elemental logical function isclose_r(a, b, atol, rtol)
        implicit none
        real (kind=kind_phys), intent(in) :: a, b
        real (kind=kind_phys), intent(in), optional :: atol, rtol

        real (kind=kind_phys) :: atol_local, rtol_local

        if (present(atol)) then
            atol_local = atol
        else
            atol_local = 1.0d-30
        end if
        if (present(rtol)) then
            rtol_local = rtol
        else
            rtol_local = 1.0d-11
        end if

        isclose_r = abs(a - b) <= (atol_local + rtol_local * abs(b))

    end function isclose_r

    elemental logical function isclose_i(a, b, atol)
        implicit none
        integer, intent(in) :: a, b
        integer, intent(in), optional :: atol

        integer :: atol_local, rtol_local

        if (present(atol)) then
            atol_local = atol
        else
            atol_local = 0
        end if

        isclose_i = abs(a - b) <= atol_local

    end function isclose_i

    subroutine tic(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t1)
    end subroutine tic

    subroutine toc(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t2)
        write(*,'(a,f5.3,a)') "    Time Taken --> ", 1000*real(t2-t1), ' ms'
    end subroutine toc

end program test
