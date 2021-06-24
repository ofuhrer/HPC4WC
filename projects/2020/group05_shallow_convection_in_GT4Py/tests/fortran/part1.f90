module mod
    integer, parameter :: kind_phys = 8

    real(kind=kind_phys),parameter:: grav   =9.80665e+0_kind_phys
    real(kind=kind_phys),parameter:: cp     =1.0046e+3_kind_phys
    real(kind=kind_phys),parameter:: hvap   =2.5000e+6_kind_phys
    real(kind=kind_phys),parameter:: rv     =4.6150e+2_kind_phys
    real(kind=kind_phys),parameter:: rd     =2.8705e+2_kind_phys
    real(kind=kind_phys),parameter:: fv     =rv/rd-1.
    real(kind=kind_phys),parameter:: t0c    =2.7315e+2_kind_phys
    real(kind=kind_phys),parameter:: cvap   =1.8460e+3_kind_phys
    real(kind=kind_phys),parameter:: cliq   =4.1855e+3_kind_phys
    real(kind=kind_phys),parameter:: eps    =rd/rv
    real(kind=kind_phys),parameter:: epsm1  =rd/rv-1.

    real(kind=kind_phys),parameter:: ttp    =2.7316e+2_kind_phys
    real(kind=kind_phys),parameter:: csol   =2.1060e+3_kind_phys
    real(kind=kind_phys),parameter:: hfus   =3.3358e+5_kind_phys
    real(kind=kind_phys),parameter:: psat   =6.1078e+2_kind_phys
contains
elemental function fpvsx(t)
    implicit none
    integer, parameter :: kind_phys = 8
    real(kind=kind_phys) fpvsx
    real(kind=kind_phys),intent(in):: t
    real(kind=kind_phys),parameter:: tliq=ttp
    real(kind=kind_phys),parameter:: tice=ttp-20.0
    real(kind=kind_phys),parameter:: dldtl=cvap-cliq
    real(kind=kind_phys),parameter:: heatl=hvap
    real(kind=kind_phys),parameter:: xponal=-dldtl/rv
    real(kind=kind_phys),parameter:: xponbl=-dldtl/rv+heatl/(rv*ttp)
    real(kind=kind_phys),parameter:: dldti=cvap-csol
    real(kind=kind_phys),parameter:: heati=hvap+hfus
    real(kind=kind_phys),parameter:: xponai=-dldti/rv
    real(kind=kind_phys),parameter:: xponbi=-dldti/rv+heati/(rv*ttp)
    real(kind=kind_phys) tr,w,pvl,pvi
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    tr=ttp/t
    if(t.ge.tliq) then
      fpvsx=psat*(tr**xponal)*exp(xponbl*(1.-tr))
    elseif(t.lt.tice) then
      fpvsx=psat*(tr**xponai)*exp(xponbi*(1.-tr))
    else
      w=(t-tice)/(tliq-tice)
      pvl=psat*(tr**xponal)*exp(xponbl*(1.-tr))
      pvi=psat*(tr**xponai)*exp(xponbi*(1.-tr))
      fpvsx=w*pvl+(1.-w)*pvi
    endif
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
end function

subroutine part1(im,ix,km,delt,itc,ntc,ntk,ntr,delp, &
     &     prslp,psp,phil,qtr,q1,t1,u1,v1,fscav, &
     &     rn,kbot,ktop,kcnv,islimsk,garea, &
     &     dot,ncloud,hpbl,ud_mf,dt_mf,cnvw,cnvc, &
     &     clam,c0s,c1,pgcon,asolfac, &
     &     heo, heso, qo, qeso)
      implicit none
!
      integer, parameter :: kind_phys = 8
      integer, intent(in)  :: im, ix,  km, itc, ntc, ntk, ntr, ncloud
      integer, intent(in)  :: islimsk(im)
      real(kind=kind_phys), intent(in) ::  delt
      real(kind=kind_phys), intent(in) :: psp(im), delp(ix,km), &
     &   prslp(ix,km), garea(im), hpbl(im), dot(ix,km), phil(ix,km)
!
      real(kind=kind_phys), intent(in) :: fscav(ntc)
      integer, intent(in)  :: kcnv(im)
      real(kind=kind_phys), intent(in) ::   qtr(ix,km,ntr+2), &
     &   q1(ix,km), t1(ix,km), u1(ix,km), v1(ix,km)
!
      integer :: kbot(im), ktop(im)
      real(kind=kind_phys) :: rn(im), &
     &   cnvw(ix,km), cnvc(ix,km), ud_mf(im,km), dt_mf(im,km)
!
      real(kind=kind_phys), intent(in) :: clam,    c0s,     c1, &
     &                                    asolfac, pgcon
!
!  local variables
      integer              i,j,indx, k, kk, km1, n
      integer              kpbl(im)
!
      real(kind=kind_phys) clamd,   tkemx,   tkemn,   dtke
!
      real(kind=kind_phys) dellat,  delta, &
     &                     c0l,     d0, &
     &                     desdt,   dp, &
     &                     dq,      dqsdp,   dqsdt,   dt, &
     &                     dt2,     dtmax,   dtmin,   dxcrt, &
     &                     dv1h,    dv2h,    dv3h, &
     &                     dv1q,    dv2q,    dv3q, &
     &                     dz,      dz1,     e1, &
     &                     el2orc,  elocp,   aafac,   cm, &
     &                     es,      etah,    h1, &
     &                     evef,    evfact,  evfactl, fact1, &
     &                     fact2,   factor,  dthk, &
     &                     g,       gamma,   pprime,  betaw, &
     &                     qlk,     qrch,    qs, &
     &                     rfact,   shear,   tfac, &
     &                     val,     val1,    val2, &
     &                     w1,      w1l,     w1s,     w2, &
     &                     w2l,     w2s,     w3,      w3l, &
     &                     w3s,     w4,      w4l,     w4s, &
     &                     rho,     tem,     tem1,    tem2,  &
     &                     ptem,    ptem1
!
      integer              kb(im), kbcon(im), kbcon1(im), &
     &                     ktcon(im), ktcon1(im), ktconn(im), &
     &                     kbm(im), kmax(im)
!
      real(kind=kind_phys) aa1(im),     cina(im),    &
     &                     tkemean(im), clamt(im), &
     &                     ps(im),      del(ix,km), prsl(ix,km), &
     &                     umean(im),   tauadv(im), gdx(im), &
     &                     delhbar(im), delq(im),   delq2(im), &
     &                     delqbar(im), delqev(im), deltbar(im), &
     &                     deltv(im),   dtconv(im), edt(im), &
     &                     pdot(im),    po(im,km), &
     &                     qcond(im),   qevap(im),  hmax(im), &
     &                     rntot(im),   vshear(im), &
     &                     xlamud(im),  xmb(im),    xmbmax(im), &
     &                     delebar(im,ntr), &
     &                     delubar(im), delvbar(im)
!
      real(kind=kind_phys) c0(im)
!
      real(kind=kind_phys) crtlamd
!
      real(kind=kind_phys) cinpcr,  cinpcrmx,  cinpcrmn, &
     &                     cinacr,  cinacrmx,  cinacrmn
!
!  parameters for updraft velocity calculation
      real(kind=kind_phys) bet1,    cd1,     f1,      gam1, &
     &                     bb1,     bb2
!    &                     bb1,     bb2,     wucb
!c
!  physical parameters
!     parameter(g=grav,asolfac=0.89)
      parameter(g=grav)
      parameter(elocp=hvap/cp, &
     &          el2orc=hvap*hvap/(rv*cp))
!     parameter(c0s=0.002,c1=5.e-4,d0=.01)
!     parameter(d0=.01)
      parameter(d0=.001)
!     parameter(c0l=c0s*asolfac)
!
! asolfac: aerosol-aware parameter based on Lim & Hong (2012)
!      asolfac= cx / c0s(=.002)
!      cx = min([-0.7 ln(Nccn) + 24]*1.e-4, c0s)
!      Nccn: CCN number concentration in cm^(-3)
!      Until a realistic Nccn is provided, Nccns are assumed
!      as Nccn=100 for sea and Nccn=1000 for land
!
      parameter(cm=1.0,delta=fv)
      parameter(fact1=(cvap-cliq)/rv,fact2=hvap/rv-fact1*t0c)
      parameter(clamd=0.1,tkemx=0.65,tkemn=0.05)
      parameter(dtke=tkemx-tkemn)
      parameter(dthk=25.)
      parameter(cinpcrmx=180.,cinpcrmn=120.)
!     parameter(cinacrmx=-120.,cinacrmn=-120.)
      parameter(cinacrmx=-120.,cinacrmn=-80.)
      parameter(crtlamd=3.e-4)
      parameter(dtmax=10800.,dtmin=600.)
      parameter(bet1=1.875,cd1=.506,f1=2.0,gam1=.5)
      parameter(betaw=.03,dxcrt=15.e3)
      parameter(h1=0.33333333)
!  local variables and arrays
      real(kind=kind_phys) pfld(im,km),    to(im,km), &    !qo(im,km), &
     &                     uo(im,km),      vo(im,km), &    !qeso(im,km), &
     &                     ctr(im,km,ntr), ctro(im,km,ntr)
!  for aerosol transport
      real(kind=kind_phys) qaero(im,km,ntc)
!  for updraft velocity calculation
      real(kind=kind_phys) wu2(im,km),     buo(im,km),    drag(im,km)
      real(kind=kind_phys) wc(im),         scaldfunc(im), sigmagfm(im)
!
!  cloud water
!     real(kind=kind_phys) qlko_ktcon(im), dellal(im,km), tvo(im,km),
      real(kind=kind_phys) qlko_ktcon(im), dellal(im,km), &
     &                     dbyo(im,km),    zo(im,km),     xlamue(im,km), &
!     &                     heo(im,km),     heso(im,km), &
     &                     dellah(im,km),  dellaq(im,km), &
     &                     dellae(im,km,ntr), &
     &                     dellau(im,km),  dellav(im,km), hcko(im,km), &
     &                     ucko(im,km),    vcko(im,km),   qcko(im,km), &
     &                     qrcko(im,km),   ecko(im,km,ntr), &
     &                     eta(im,km), &
     &                     zi(im,km),      pwo(im,km),    c0t(im,km), &
     &                     sumx(im),       tx1(im),       cnvwt(im,km)
!
      logical do_aerosols, totflg, cnvflg(im), flg(im)
      real(kind=kind_phys), intent(out) :: heo(im,km), heso(im,km), qo(im,km), qeso(im,km)
!
      real(kind=kind_phys) tf, tcr, tcrf
      parameter (tf=233.16, tcr=263.16, tcrf=1.0/(tcr-tf))
!
!-----------------------------------------------------------------------
!>  ## Determine whether to perform aerosol transport
      do_aerosols = (itc > 0) .and. (ntc > 0) .and. (ntr > 0)
      if (do_aerosols) do_aerosols = (ntr >= itc + ntc - 3)
!
!************************************************************************
!     convert input Pa terms to Cb terms  -- Moorthi
!>  ## Compute preliminary quantities needed for the static and feedback control portions of the algorithm.
!>  - Convert input pressure terms to centibar units.
      ps   = psp   * 0.001
      prsl = prslp * 0.001
      del  = delp  * 0.001
!************************************************************************
!
      km1 = km - 1
!
!  initialize arrays
!
!>  - Initialize column-integrated and other single-value-per-column variable arrays.
      do i=1,im
        cnvflg(i) = .true.
        if(kcnv(i) == 1) cnvflg(i) = .false.
        if(cnvflg(i)) then
          kbot(i)=km+1
          ktop(i)=0
        endif
        rn(i)=0.
        kbcon(i)=km
        ktcon(i)=1
        ktconn(i)=1
        kb(i)=km
        pdot(i) = 0.
        qlko_ktcon(i) = 0.
        edt(i)  = 0.
        aa1(i)  = 0.
        cina(i) = 0.
        vshear(i) = 0.
        gdx(i) = sqrt(garea(i))
      enddo
!!
!>  - Return to the calling routine if deep convection is present or the surface buoyancy flux is negative.
      totflg = .true.
      do i=1,im
        totflg = totflg .and. (.not. cnvflg(i))
      enddo
      if(totflg) return
!!
!>  - determine aerosol-aware rain conversion parameter over land
      do i=1,im
        if(islimsk(i) == 1) then
           c0(i) = c0s*asolfac
        else
           c0(i) = c0s
        endif
      enddo
!
!>  - determine rain conversion parameter above the freezing level which exponentially decreases with decreasing temperature from Han et al.'s (2017) \cite han_et_al_2017 equation 8.
      do k = 1, km
        do i = 1, im
          if(t1(i,k) > 273.16) then
            c0t(i,k) = c0(i)
          else
            tem = d0 * (t1(i,k) - 273.16)
            tem1 = exp(tem)
            c0t(i,k) = c0(i) * tem1
          endif
        enddo
      enddo
!
!>  - Initialize convective cloud water and cloud cover to zero.
      do k = 1, km
        do i = 1, im
          cnvw(i,k) = 0.
          cnvc(i,k) = 0.
        enddo
      enddo
! hchuang code change
!>  - Initialize updraft mass fluxes to zero.
      do k = 1, km
        do i = 1, im
          ud_mf(i,k) = 0.
          dt_mf(i,k) = 0.
        enddo
      enddo
!
      dt2   = delt
!
!  model tunable parameters are all here
!     clam    = .3
!     aafac   = .1
      aafac   = .05
!     evef    = 0.07
      evfact  = 0.3
      evfactl = 0.3
!
!     pgcon   = 0.7     ! Gregory et al. (1997, QJRMS)
!     pgcon   = 0.55    ! Zhang & Wu (2003,JAS)
      w1l     = -8.e-3
      w2l     = -4.e-2
      w3l     = -5.e-3
      w4l     = -5.e-4
      w1s     = -2.e-4
      w2s     = -2.e-3
      w3s     = -1.e-3
      w4s     = -2.e-5
!
!  define top layer for search of the downdraft originating layer
!  and the maximum thetae for updraft
!
!>  - Determine maximum indices for the parcel starting point (kbm) and cloud top (kmax).
      do i=1,im
        kbm(i)   = km
        kmax(i)  = km
        tx1(i)   = 1.0 / ps(i)
      enddo
!
      do k = 1, km
        do i=1,im
          if (prsl(i,k)*tx1(i) > 0.70) kbm(i)   = k + 1
          if (prsl(i,k)*tx1(i) > 0.60) kmax(i)  = k + 1
        enddo
      enddo
      do i=1,im
        kbm(i)   = min(kbm(i),kmax(i))
      enddo
!
!  hydrostatic height assume zero terr and compute
!  updraft entrainment rate as an inverse function of height
!
!>  - Calculate hydrostatic height at layer centers assuming a flat surface (no terrain) from the geopotential.
      do k = 1, km
        do i=1,im
          zo(i,k) = phil(i,k) / g
        enddo
      enddo
!>  - Calculate interface height
      do k = 1, km1
        do i=1,im
          zi(i,k) = 0.5*(zo(i,k)+zo(i,k+1))
        enddo
      enddo
!
!  pbl height
!
!>  - Find the index for the PBL top using the PBL height; enforce that it is lower than the maximum parcel starting level.
      do i=1,im
        flg(i) = cnvflg(i)
        kpbl(i)= 1
      enddo
      do k = 2, km1
        do i=1,im
          if (flg(i) .and. zo(i,k) <= hpbl(i)) then
            kpbl(i) = k
          else
            flg(i) = .false.
          endif
        enddo
      enddo
      do i=1,im
        kpbl(i)= min(kpbl(i),kbm(i))
      enddo
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!   convert surface pressure to mb from cb
!
!>  - Convert prsl from centibar to millibar, set normalized mass flux to 1, cloud properties to 0, and save model state variables (after advection/turbulence).
      do k = 1, km
        do i = 1, im
          if (cnvflg(i) .and. k <= kmax(i)) then
            pfld(i,k) = prsl(i,k) * 10.0
            eta(i,k)  = 1.
            hcko(i,k) = 0.
            qcko(i,k) = 0.
            qrcko(i,k)= 0.
            ucko(i,k) = 0.
            vcko(i,k) = 0.
            dbyo(i,k) = 0.
            pwo(i,k)  = 0.
            dellal(i,k) = 0.
            to(i,k)   = t1(i,k)
            qo(i,k)   = q1(i,k)
            uo(i,k)   = u1(i,k)
            vo(i,k)   = v1(i,k)
!           uo(i,k)   = u1(i,k) * rcs(i)
!           vo(i,k)   = v1(i,k) * rcs(i)
            wu2(i,k)  = 0.
            buo(i,k)  = 0.
            drag(i,k) = 0.
            cnvwt(i,k) = 0.
          endif
        enddo
      enddo
!
!  initialize tracer variables
!
      do n = 3, ntr+2
        kk = n-2
        do k = 1, km
          do i = 1, im
            if (cnvflg(i) .and. k <= kmax(i)) then
              ctr(i,k,kk)  = qtr(i,k,n)
              ctro(i,k,kk) = qtr(i,k,n)
              ecko(i,k,kk) = 0.
            endif
          enddo
        enddo
      enddo
!>  - Calculate saturation specific humidity and enforce minimum moisture values.
      do k = 1, km
        do i=1,im
          if (cnvflg(i) .and. k <= kmax(i)) then
            qeso(i,k) = 0.01 * fpvsx(to(i,k))      ! fpvs is in pa
            qeso(i,k) = eps * qeso(i,k) / (pfld(i,k) + epsm1*qeso(i,k))
            val1      =             1.e-8
            qeso(i,k) = max(qeso(i,k), val1)
            val2      =           1.e-10
            qo(i,k)   = max(qo(i,k), val2 )
!           qo(i,k)   = min(qo(i,k),qeso(i,k))
!           tvo(i,k)  = to(i,k) + delta * to(i,k) * qo(i,k)
          endif
        enddo
      enddo
!
!  compute moist static energy
!
!>  - Calculate moist static energy (heo) and saturation moist static energy (heso).
      do k = 1, km
        do i=1,im
          if (cnvflg(i) .and. k <= kmax(i)) then
!           tem       = g * zo(i,k) + cp * to(i,k)
            tem       = phil(i,k) + cp * to(i,k)
            heo(i,k)  = tem  + hvap * qo(i,k)
            heso(i,k) = tem  + hvap * qeso(i,k)
!           heo(i,k)  = min(heo(i,k),heso(i,k))
          endif
        enddo
      enddo
end subroutine part1

end module mod