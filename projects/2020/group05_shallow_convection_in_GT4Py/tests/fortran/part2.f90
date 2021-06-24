module mod
  ! PARAMETERS
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
  ! INPUT VALUES
  integer im, ix, km
  real(kind=kind_phys) clam, c0s, c1, asolfac, pgcon
  ! INTERMEDIATE ARRAYS
  logical cnvflg(ix)
  integer kb(im,km), kmax(im,km)
  real(kind=kind_phys) heo(im,km), heso(im,km)
  real(kind=kind_phys) zi(im,km), xlamue(im,km), xlamud(im)
  real(kind=kind_phys) hcko(im,km), dbyo(im,km), ucko(im,km), vcko(im,km)
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

subroutine part2_init(im_v, ix_v, km_v, clam_v, c0s_v, c1_v, asolfac_v, pgcon_v, &
                     &cnvflg_v, kb_v, kmax_v, heo_v, heso_v, zi_v, xlamue_v, xlamud_v, &
                     &hcko_v, dbyo_v, ucko_v, vcko_v)
  implicit none
  integer, intent(in):: im_v, ix_v, km_v
  real(kind=kind_phys), intent(in):: clam_v, c0s_v, c1_v, asolfac_v, pgcon_v
  ! INTERMEDIATE ARRAYS
  logical, intent(in):: cnvflg_v(ix_v)
  integer, intent(in):: kb_v(im_v,km_v), kmax_v(im_v,km_v)
  real(kind=kind_phys), intent(in):: heo_v(im_v,km_v), heso_v(im_v,km_v)
  real(kind=kind_phys), intent(in):: zi_v(im_v,km_v), xlamue_v(im_v,km_v), xlamud_v(im_v)
  real(kind=kind_phys), intent(in):: hcko_v(im_v,km_v), dbyo_v(im_v,km_v), ucko_v(im_v,km_v), vcko_v(im_v,km_v)
  im = im_v
  ix = ix_v
  km = km_v
  clam = clam_v
  c0s = c0s_v
  c1 = c1_v
  asolfac = asolfac_v
  pgcon = pgcon_v
  do i = 1, im
    cnvflg(i) = cnvflg_v(i)
    xlamud(i) = xlamud_v(i)
    do k = 1, km
      kb(i,k) = kb_v(i,k)
      kmax(i,k) = kmax_v(i,k)
      heo(i,k) = heo_v(i,k)
      heso(i,k) = heso_v(i,k)
      zi(i,k) = zi_v(i,k)
      xlamue(i,k) = xlamue_v(i,k)
      hcko(i,k) = hcko_v(i,k)
      dbyo(i,k) = dbyo_v(i,k)
      ucko(i,k) = ucko_v(i,k)
      vcko(i,k) = vcko_v(i,k)
    end do
  end do



end subroutine part2_init

subroutine part2_stencil7_line756(hcko, dbyo, ucko, vcko)
  implicit none
  integer km1
  km1 = km - 1
!
!  cm is an enhancement factor in entrainment rates for momentum
!
!> - Calculate the cloud properties as a parcel ascends, modified by entrainment and detrainment. Discretization follows Appendix B of Grell (1993) \cite grell_1993 . Following Han and Pan (2006) \cite han_and_pan_2006, the convective momentum transport is reduced by the convection-induced pressure gradient force by the constant "pgcon", currently set to 0.55 after Zhang and Wu (2003) \cite zhang_and_wu_2003 .
  do k = 2, km1
    do i = 1, im
      if (cnvflg(i)) then
        if(k > kb(i) .and. k < kmax(i)) then
          dz   = zi(i,k) - zi(i,k-1)
          tem  = 0.5 * (xlamue(i,k)+xlamue(i,k-1)) * dz
          tem1 = 0.5 * xlamud(i) * dz
          factor = 1. + tem - tem1
          hcko(i,k) = ((1.-tem1)*hcko(i,k-1)+tem*0.5* &
 &                     (heo(i,k)+heo(i,k-1)))/factor
          dbyo(i,k) = hcko(i,k) - heso(i,k)
!
          tem  = 0.5 * cm * tem
          factor = 1. + tem
          ptem = tem + pgcon
          ptem1= tem - pgcon
          ucko(i,k) = ((1.-tem)*ucko(i,k-1)+ptem*uo(i,k) &
 &                     +ptem1*uo(i,k-1))/factor
          vcko(i,k) = ((1.-tem)*vcko(i,k-1)+ptem*vo(i,k) &
 &                     +ptem1*vo(i,k-1))/factor
        endif
      endif
    enddo
  enddo
end subroutine part2_stencil7_line756
end module mod