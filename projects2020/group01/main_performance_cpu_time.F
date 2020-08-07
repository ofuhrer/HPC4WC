program test 
     use m_countdown, only : cdstart, cdshow, cdstop
     use machine, only : kind_phys
     use funcphys, only : gfuncphys
     use module_sfc_sice, only : sfc_sice
     implicit none

     integer:: i,j,k,l,im, number_ice_grids, ipr, km, iterations
     integer, dimension(11):: grid_init
     integer, dimension(:), allocatable :: islimsk, ones

     real, dimension(11):: sea_ice_perc
     real (kind=kind_phys) :: delt, cimin, ps_l, t1_l, q1_l, delt_l, sfcemis_l,           &
         dlwflx_l,sfcnsw_l, sfcdsw_l, srflag_l, cm_l,      &
         ch_l, prsl1_l, prslki_l, islimsk_l, wind_l,              &
         cimin_l, hice_l, fice_l, tice_l,            &
         weasd_l, tskin_l, tprcp_l, stc_l, ep_l, snwdph_l,    &
         qsurf_l, snowmt_l, gflux_l, cmm_l, chh_l, evap_l,       &
         hflx_l, ps_i, t1_i, q1_i, delt_i, sfcemis_i,           &
         dlwflx_i,sfcnsw_i, sfcdsw_i, srflag_i, cm_i,      &
         ch_i, prsl1_i, prslki_i, islimsk_i, wind_i,              &
         cimin_i, hice_i, fice_i, tice_i,            &
         weasd_i, tskin_i, tprcp_i, stc_i, ep_i, snwdph_i,    &
         qsurf_i, snowmt_i, gflux_i, cmm_i, chh_i, evap_i,       &
         hflx_i

     real (kind=kind_phys), dimension(:, :), allocatable :: stc
     real (kind=kind_phys), dimension(:), allocatable :: ps, t1, q1, &
         sfcemis, dlwflx, sfcnsw, sfcdsw, srflag, cm, ch, prsl1, prslki, &
         wind, hice, fice, tice, weasd, tskin, tprcp, ep, snwdph, &
         qsurf, snowmt, gflux, cmm, chh, evap, hflx
     real(8), dimension(:),allocatable :: time1, time2, timediff
    
     logical, dimension(:), allocatable :: flag_iter, flag
     logical:: flag_iter_l, flag_iter_i,lprnt

     ! define number of interations
     iterations = 10

     ! initialise arrays for loop   
     grid_init = (/32,64,128,256,512,1024,2048,4096,8192,16384,32768/)      
     sea_ice_perc =(/0., 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,1./)  

     ! allocate time arrays
     allocate(time1(iterations),time2(iterations)) 
 
     ! initialize
     call gfuncphys()

     ! read land data
     open(1,file= 'land.dat', status='old')
     read(1,*) ps_l, t1_l, q1_l, delt_l, sfcemis_l,           &
     &    dlwflx_l,sfcnsw_l, sfcdsw_l, srflag_l, cm_l,      &
     &    ch_l, prsl1_l, prslki_l, islimsk_l, wind_l,              &
     &    flag_iter_l,cimin_l, hice_l, fice_l, tice_l,            &
     &    weasd_l, tskin_l, tprcp_l, stc_l, ep_l, snwdph_l,    &
     &    qsurf_l, snowmt_l, gflux_l, cmm_l, chh_l, evap_l,       &
     &    hflx_l
     close(1)

     ! read ice data
     open(2,file= 'ice.dat', status='old')
     read(2,*) ps_i, t1_i, q1_i, delt_i, sfcemis_i,           &
     &    dlwflx_i,sfcnsw_i, sfcdsw_i, srflag_i, cm_i,      &
     &    ch_i, prsl1_i, prslki_i, islimsk_i, wind_i,              &
     &    flag_iter_i,cimin_i, hice_i, fice_i, tice_i,            &
     &    weasd_i, tskin_i, tprcp_i, stc_i, ep_i, snwdph_i,    &
     &    qsurf_i, snowmt_i, gflux_i, cmm_i, chh_i, evap_i,       &
     &    hflx_i
     close(2)

     ! define ipr, km and lprnt
     ipr = 10
     lprnt = .false.
     km = 4
     
     ! loop over gridpoint numbers
     open(3,file = 'results_cpu_time.dat', status = 'replace')
     do k = 1,11
     
         im = grid_init(k)
     
         allocate(ps(im), t1(im), q1(im), sfcemis(im), dlwflx(im), sfcnsw(im), &
             sfcdsw(im), srflag(im), cm(im), ch(im), prsl1(im), prslki(im), &
             wind(im), hice(im), fice(im), tice(im), weasd(im), tskin(im), &
             tprcp(im), ep(im), snwdph(im), qsurf(im), snowmt(im), gflux(im), &
             cmm(im), chh(im), evap(im), hflx(im), stc(im, km), &
             islimsk(im),flag_iter(im), &
             ones(im), flag(im))
          
         ones(:) = 1

         ! loop percentage
         do j = 1,11

            number_ice_grids =int(im *sea_ice_perc(j))

            ps(1:number_ice_grids) = ps_i  
            ps(number_ice_grids+1:im) = ps_l
            t1(1:number_ice_grids) = t1_i 
            t1(number_ice_grids+1:im) = t1_l
            q1(1:number_ice_grids)  = q1_i
            q1(number_ice_grids+1:im) = q1_l
            sfcemis(1:number_ice_grids) = sfcemis_i
            sfcemis(number_ice_grids+1:im) = sfcemis_l
            dlwflx(1:number_ice_grids) = dlwflx_i
            dlwflx(number_ice_grids+1:im) = dlwflx_l
            sfcnsw(1:number_ice_grids) = sfcnsw_i
            sfcnsw(number_ice_grids+1:im) = sfcnsw_l
            sfcdsw(1:number_ice_grids) = sfcdsw_i
            sfcdsw(number_ice_grids+1:im) = sfcdsw_l
            srflag(1:number_ice_grids) = srflag_i
            srflag(number_ice_grids+1:im) = srflag_l
            cm(1:number_ice_grids) = cm_i
            cm(number_ice_grids+1:im) = cm_l
            ch(1:number_ice_grids) = ch_i
            ch(number_ice_grids+1:im) = ch_l
            prsl1(1:number_ice_grids) = prsl1_i
            prsl1(number_ice_grids+1:im) = prsl1_l
            prslki(1:number_ice_grids) = prslki_i
            prslki(number_ice_grids+1:im) = prslki_l
            wind(1:number_ice_grids) = wind_i
            wind(number_ice_grids+1:im) = wind_l
            hice(1:number_ice_grids) = hice_i
            hice(number_ice_grids+1:im) = hice_l
            fice(1:number_ice_grids) = fice_i
            fice(number_ice_grids+1:im) = hice_l
            tice(1:number_ice_grids) = tice_i
            tice(number_ice_grids+1:im) = tice_l
            weasd(1:number_ice_grids) = weasd_i
            weasd(number_ice_grids+1:im) = weasd_l
            tskin(1:number_ice_grids) = tskin_i
            tskin(number_ice_grids+1:im) = tskin_l
            tprcp(1:number_ice_grids) = tprcp_i
            tprcp(number_ice_grids+1:im) = tprcp_l
            ep(1:number_ice_grids) = ep_i
            ep(number_ice_grids+1:im) = ep_l
            snwdph(1:number_ice_grids) = snwdph_i
            snwdph(number_ice_grids+1:im) = snwdph_l
            qsurf(1:number_ice_grids) = qsurf_i
            qsurf(number_ice_grids+1:im) = qsurf_l
            snowmt(1:number_ice_grids) = snowmt_i
            snowmt(number_ice_grids+1:im) = snowmt_l
            gflux(1:number_ice_grids) = gflux_i
            gflux(number_ice_grids+1:im) = gflux_l
            cmm(1:number_ice_grids) = cmm_i
            cmm(number_ice_grids+1:im) = cmm_l
            chh(1:number_ice_grids) = chh_i
            chh(number_ice_grids+1:im) = chh_l
            evap(1:number_ice_grids) = evap_i
            evap(number_ice_grids+1:im) = evap_l
            hflx(1:number_ice_grids) = hflx_i
            hflx(number_ice_grids+1:im) = hflx_l
            stc(1:number_ice_grids,:) = stc_i
            stc(number_ice_grids+1:im,:) = stc_l
            islimsk(1:number_ice_grids) = islimsk_i
            islimsk(number_ice_grids+1:im) = islimsk_l
            flag_iter(1:number_ice_grids) = flag_iter_i
            flag_iter(number_ice_grids+1:im) = flag_iter_l
            

            ! iteration loop
            do l = 1,iterations
                call tic(time1(l), time2(l))

                call sfc_sice                                              &
                !  ---  inputs:
                ( im, km, ps, t1, q1, delt,                                &
                sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,                   &
                cm, ch, prsl1, prslki, islimsk, wind,                      &
                flag_iter, lprnt, ipr, cimin,                              &
                !  ---  input/outputs:
                hice, fice, tice, weasd, tskin, tprcp, stc, ep,            &
                snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx )


                call toc(time1(l), time2(l))
                
                ! write gridpoint, percentage, iteration
                write(*,'(I5)', advance="no")  im 
                write(*,'(F10.2)', advance="no")  sea_ice_perc(j)
                write(*,'(I5)', advance="no")  l

                ! write in file   
                write(3,'(I5,I6,F10.2,F10.3)') l,im, sea_ice_perc(j), 1000*real(time2(l)-time1(l))

               
            enddo
            
         enddo
 
         deallocate(ps, t1, q1, sfcemis, dlwflx, sfcnsw, &
             sfcdsw, srflag, cm, ch, prsl1, prslki, &
             wind, hice, fice, tice, weasd, tskin, &
             tprcp, ep, snwdph, qsurf, snowmt, gflux, &
             cmm, chh, evap, hflx, stc, &
             islimsk,flag_iter,ones, flag)
     enddo   
 
     close(3)

     deallocate(time1,time2) 

contains 

    subroutine tic(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t1)
    end subroutine tic

    subroutine toc(t1, t2)
        implicit none
        real(8) :: t1, t2
        call cpu_time(t2)
        write(*,'(a,f10.3,a)') "    Time Taken --> ", 1000*real(t2-t1), ' ms'
    end subroutine toc



end  program      
                  
                  
