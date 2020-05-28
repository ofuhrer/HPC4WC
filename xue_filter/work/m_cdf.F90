! Module:
! m_cdf
!
! Author:
! Oliver Fuhrer, Tue Jun 27 08:21:28 CEST 2006
! oliver.fuhrer@epfl.ch
!
! Description:
! This module contains a collection of subroutines
! and functions for reading and writing data as well
! as handling NetCDF files. This module is base on the
! Fortran90 NetCDF interface. Function overloading is
! used to make usage for 0d,1d,2d,3d variables trans-
! parent. Fields can be time dependent. The order of
! space dimensions can be flipped for storage if re-
! quired via an option. COARDS standard is observerd
! and scaled/offsetted data will be handled correctly.
! Ideally, calls are made using keyword arguments.
!
! Example (how to write a simple NetCDF file):
!
!  ! begin session
!  call cdfbegin(fname,'example.nc',.true.)
!  ! define time axis
!  call cdftime('time','seconds since 1975-17-12 03:00:00 LT')
!  ! dimensions
!  call cdfdimput('x',nx)
!  call cdfdimput('z',nz)
!  ! attributes
!  call cdfattput('','Conventions','COARDS')
!  call cdfattput('','Source','example')
!  call cdfattput('','dx',dx)
!  call cdfattput('','dz',dz)
!  ! time independent variables
!  call cdfvarput('x',x(1:nx),(/'x'/),units='m')
!  call cdfvarput('z',z(1:nz),(/'z'/),units='m')
!  call cdfvarput('xc',xc(1:nx,1:nz),(/'x','z'/),units='m')
!  call cdfvarput('zc',zc(1:nx,1:nz),(/'x','z'/),units='m')
!  ! time dependent variables
!  call cdfvarput('cfl',cfl,time=t)
!  call cdfvarput('rho',rho(1:nx,1:nz),(/'x','z'/),time=t,units='kg/m3')
!  call cdfvarput('u',u(1:nx,1:nz),(/'x','z'/),time=t,units='m/s')
!  call cdfvarput('w',w(1:nx,1:nz),(/'x','z'/),time=t,units='m/s')
!  call cdfvarput('theta',theta(1:nx,1:nz),(/'x','z'/),time=t,units='K')
!  ! end session
!  call cdfend()
!
! Version History
! 02.04.2007 OF
!            Major changes! Merged time-dependent and time-independent
!            routines into one. Change philosophy to keyworded arguments.
!            Introduced possibility to read/write subdomains. Introduced
!            automatic handling of COARDS standard. As well as some bugfixes.
! 30.03.2007 OF
!            Some minor bugfixing due to bugs discovered using
!            the test routine test_cdf.f90
! 21.03.2007 OF
!            Addition of routines to inquire about presence
!            of dimensions, variables and attributes.
! 20.03.2007 OF
!            Simple modifications made for reading of NCEP reanalysis
!            data, since we had the wrong name for the units attribute.
!            Error fixed in the cdftimiget routine.
! 27.06.2006 OF
!            First version implementing reading and writing
!            fields up to three dimensions including a time
!            dimension. This version is used mainly for writing
!            and reading simples 2d computational meshes and
!            writing some simple 2d fields into NetCDF files.
!            Rigorous debuggin still to be done...

module m_cdf
#ifdef USE_IO
  use netcdf
  use m_constants, only : wp, one, zero, stdout
  implicit none
  private

  ! write debuggin information?
  logical, parameter :: cdf_debug = .false.

  ! private module variables
  logical :: cdf_active = .false.       ! is a session open with caller
  character(len=256) :: cdf_fname       ! file name to work on
  character(len=256) :: cdf_caller      ! name of calling routine
  logical :: cdf_tactive = .false.      ! is time dimension active?
  character(len=256) :: cdf_dtime = ''  ! name of time dimension
  character(len=256) :: cdf_dtimeu = '' ! unit of time dimension
  integer :: cdf_dtimeid                ! id of time dimension
  integer :: cdf_ncid                   ! id number of file

  ! overloaded public functions
  interface cdfvarput
     module procedure cdfvarr3put,cdfvarr2put,cdfvarr1put,cdfvarr0put
     module procedure cdfvari1put,cdfvari0put
  end interface
  interface cdfvarget
     module procedure cdfvarr3get,cdfvarr2get,cdfvarr1get,cdfvarr0get
     module procedure cdfvari1get,cdfvari0get
  end interface
  interface cdfattput
     module procedure cdfattrput,cdfattiput,cdfatttput
  end interface
  interface cdfattget
     module procedure cdfattrget,cdfatttget,cdfattiget
  end interface

  ! overloaded private functions
  interface cdfcopy
     module procedure cdfcopyr3,cdfcopyr2,cdfcopyr1,cdfcopyr0
     module procedure cdfcopyi3,cdfcopyi2,cdfcopyi1,cdfcopyi0
  end interface

  ! visibility of functions
  public :: cdfbegin,cdfend,cdftime,cdftimeget, &
       cdfdimyes,cdfdimput,cdfdimget, &
       cdfattyes,cdfattput,cdfattget, &
       cdfvaryes,cdfvarput,cdfvarget

contains


  !**************************************
  ! is a dimension present?
  function cdfdimyes(name)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    
    ! out
    logical :: cdfdimyes

    ! local
    integer :: dimid, ierr

    call cdfdebug('cdfdimyes: entry '//trim(name))

    ! inquire about dimension
    ierr=nf90_inq_dimid(cdf_ncid,name,dimid)
    if (ierr.eq.nf90_noerr) then
       cdfdimyes=.true.
    else
       cdfdimyes=.false.
    end if

    call cdfdebug('cdfdimyes: exit')
    
  end function cdfdimyes


  !**************************************
  ! is a variable present?
  function cdfvaryes(name)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    
    ! out
    logical :: cdfvaryes

    ! local
    integer :: varid, ierr

    call cdfdebug('cdfvaryes: entry '//trim(name))

    ! inquire about variable
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.nf90_noerr) then
       cdfvaryes=.true.
    else
       cdfvaryes=.false.
    end if

    call cdfdebug('cdfvaryes: exit')
    
  end function cdfvaryes


  !**************************************
  ! is an attribute present?
  function cdfattyes(var,name)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var
    character(len=*), intent(in) :: name
    
    ! out
    logical :: cdfattyes

    ! local
    integer :: varid,xtype,len,attnum,ierr

    call cdfdebug('cdfattyes: entry '//trim(var)//' '//trim(name))

    ! inquire about variable and attribute
    cdfattyes=.false.
    if (var.ne.'') then
       ierr=nf90_inq_varid(cdf_ncid,var,varid)
    else
       varid=NF90_GLOBAL
       ierr=0
    end if
    if (ierr.eq.nf90_noerr) then
       ierr=nf90_inquire_attribute(cdf_ncid,varid,name,xtype,len,attnum)
       if (ierr.eq.nf90_noerr) then
          cdfattyes=.true.
       end if
    end if

    call cdfdebug('cdfattyes: exit')
    
  end function cdfattyes


  !**************************************
  ! write a time dependent variable of real kind (3d)
  subroutine cdfvarr3put(name,val,dims,flip,time,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    real (kind=wp), intent(in) :: val(:,:,:)
    character(len=*), intent(in) :: dims(:)
    logical, intent(in), optional :: flip
    real (kind=wp), intent(in), optional :: time
    integer, optional :: slice(:,:)
    character(len=*), intent(in), optional :: units
    character(len=*), intent(in), optional :: long_name    
    real (kind=wp), intent(in),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr3put'
    real (kind=wp), allocatable :: r(:,:,:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),itmp(nf90_max_dims)
    integer :: rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=3
    if (present(time)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! check space dimensions
    if (nds.gt.0) then
       call cdferror(size(dims).ne.nds, &
            subname//': number of dimension names specified incorrect')
       do i=1,nds
          call cdferr(nf90_inq_dimid(cdf_ncid,dims(i),dimids(i)))
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       call cdfinvertvec(dimids(1:nds),flip=lflip)
       call cdfinvertvec(rlen(1:nds),flip=lflip)
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
            subname//': space dimensions in file and variable not consistent')
    end if

    ! time handling
    if (present(time)) then
       call cdftimeput(it,time)
       dimids(nd)=cdf_dtimeid
    end if

    ! check if variable already present
    !   if yes: check dimensions
    !   if no:  define
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,itmp,natts))
       call cdferror(ndims.ne.nd, &
            subname//': variable already in file of different number of dimensions')
       if (nd.gt.0) &
            call cdferror(.not.cdfcheckdim(dimids(1:nd),itmp(1:nd)), &
            subname//': variable already in file of different dimension size')
    else
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_var(cdf_ncid,name,nf90_real,dimids(1:nd),varid))
       call cdferr(nf90_enddef(cdf_ncid))
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time)) then
       lstart(nd)=it
    end if

    ! allocate memory
    allocate(r(lcount(1),lcount(2),lcount(3),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')
    call cdfcopy(r(:,:,:,1),val,flip=lflip)
    
    ! write variable
    if (present(time)) then
       call cdferr(nf90_put_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_put_var(cdf_ncid,varid,r(:,:,:,1),lstart,lcount,lstride))
    end if

    ! put attributes
    if (present(units)) &
         call cdfattput(name,'units',units)
    if (present(long_name)) &
         call cdfattput(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattput(name,'missing_value',missing_value)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr3put


  !**************************************
  ! write a time dependent variable of real kind (2d)
  subroutine cdfvarr2put(name,val,dims,flip,time,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    real (kind=wp), intent(in) :: val(:,:)
    character(len=*), intent(in) :: dims(:)
    logical, intent(in), optional :: flip
    real (kind=wp), intent(in), optional :: time
    integer, optional :: slice(:,:)
    character(len=*), intent(in), optional :: units
    character(len=*), intent(in), optional :: long_name    
    real (kind=wp), intent(in),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr2put'
    real (kind=wp), allocatable :: r(:,:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),itmp(nf90_max_dims)
    integer :: rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=2
    if (present(time)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! check space dimensions
    if (nds.gt.0) then
       call cdferror(size(dims).ne.nds, &
            subname//': number of dimension names specified incorrect')
       do i=1,nds
          call cdferr(nf90_inq_dimid(cdf_ncid,dims(i),dimids(i)))
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       call cdfinvertvec(dimids(1:nds),flip=lflip)
       call cdfinvertvec(rlen(1:nds),flip=lflip)
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
            subname//': dimensions in file and variable not consistent')
    end if
    
    ! time handling
    if (present(time)) then
       call cdftimeput(it,time)
       dimids(nd)=cdf_dtimeid
    end if

    ! check if variable already present
    !   if yes: check dimensions
    !   if no:  define
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,itmp,natts))
       call cdferror(ndims.ne.nd, &
            subname//': variable already in file of different number of dimensions')
       if (nd.gt.0) &
            call cdferror(.not.cdfcheckdim(dimids(1:nd),itmp(1:nd)), &
            subname//': variable already in file of different dimension size')
    else
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_var(cdf_ncid,name,nf90_real,dimids(1:nd),varid))
       call cdferr(nf90_enddef(cdf_ncid))
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time)) then
       lstart(nd)=it
    end if

    ! allocate memory
    allocate(r(lcount(1),lcount(2),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')
    call cdfcopy(r(:,:,1),val,flip=lflip)
    
    ! write variable
    if (present(time)) then
       call cdferr(nf90_put_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_put_var(cdf_ncid,varid,r(:,:,1),lstart,lcount,lstride))
    end if

    ! put attributes
    if (present(units)) &
         call cdfattput(name,'units',units)
    if (present(long_name)) &
         call cdfattput(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattput(name,'missing_value',missing_value)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr2put


  !**************************************
  ! write a time dependent variable of real kind (1d)
  subroutine cdfvarr1put(name,val,dims,flip,time,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    real (kind=wp), intent(in) :: val(:)
    character(len=*), intent(in) :: dims(:)
    logical, intent(in), optional :: flip
    real (kind=wp), intent(in),optional :: time
    integer, optional :: slice(:,:)
    character(len=*), intent(in),optional :: units
    character(len=*), intent(in), optional :: long_name    
    real (kind=wp), intent(in),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr1put'
    real (kind=wp), allocatable :: r(:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),itmp(nf90_max_dims)
    integer :: rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=1
    if (present(time)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! check space dimensions
    if (nds.gt.0) then
       call cdferror(size(dims).ne.nds, &
            subname//': number of dimension names specified incorrect')
       do i=1,nds
          call cdferr(nf90_inq_dimid(cdf_ncid,dims(i),dimids(i)))
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       call cdfinvertvec(dimids(1:nds),flip=lflip)
       call cdfinvertvec(rlen(1:nds),flip=lflip)
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
            subname//': dimensions in file and variable not consistent')
    end if

    ! time handling
    if (present(time)) then
       call cdftimeput(it,time)
       dimids(nd)=cdf_dtimeid
    end if


    ! check if variable already present
    !   if yes: check dimensions
    !   if no:  define
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,itmp,natts))
       call cdferror(ndims.ne.nd, &
            subname//': variable already in file of different number of dimensions')
       if (nd.gt.0) &
            call cdferror(.not.cdfcheckdim(dimids(1:nd),itmp(1:nd)), &
            subname//': variable already in file of different dimension size')
    else
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_var(cdf_ncid,name,nf90_real,dimids(1:nd),varid))
       call cdferr(nf90_enddef(cdf_ncid))
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time)) then
       lstart(nd)=it
    end if

    ! allocate memory
    allocate(r(lcount(1),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')
    call cdfcopy(r(:,1),val,flip=lflip)

    ! write variable
    if (present(time)) then
       call cdferr(nf90_put_var(cdf_ncid,varid,r(:,1),lstart,lcount,lstride))
    else
       call cdferr(nf90_put_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    end if

    ! put attributes
    if (present(units)) &
         call cdfattput(name,'units',units)
    if (present(long_name)) &
         call cdfattput(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattput(name,'missing_value',missing_value)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr1put


  !**************************************
  ! write a time dependent variable of real kind (0d)
  subroutine cdfvarr0put(name,val,dims,flip,time,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    real (kind=wp), intent(in) :: val
    character(len=*), intent(in) :: dims(:)
    logical, intent(in), optional :: flip
    real (kind=wp), intent(in), optional :: time
    integer, optional :: slice(:,:)
    character(len=*), intent(in), optional :: units
    character(len=*), intent(in), optional :: long_name    
    real (kind=wp), intent(in),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr0put'
    real (kind=wp), allocatable :: r(:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),itmp(nf90_max_dims)
    integer :: rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=0
    if (present(time)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! check space dimensions
    if (nds.gt.0) then
       call cdferror(size(dims).ne.nds, &
            subname//': number of dimension names specified incorrect')
       do i=1,nds
          call cdferr(nf90_inq_dimid(cdf_ncid,dims(i),dimids(i)))
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
            subname//': space dimensions in file and variable not consistent')
    end if

    ! time handling
    if (present(time)) then
       call cdftimeput(it,time)
       dimids(nd)=cdf_dtimeid
    end if

    ! check if variable already present
    !   if yes: check dimensions
    !   if no:  define
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,itmp,natts))
       call cdferror(ndims.ne.nd, &
            subname//': variable already in file of different number of dimensions')
       if (nd.gt.0) &
            call cdferror(.not.cdfcheckdim(dimids(1:nd),itmp(1:nd)), &
            subname//': variable already in file of different dimension size')
    else
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_var(cdf_ncid,name,nf90_real,dimids(1:nd),varid))
       call cdferr(nf90_enddef(cdf_ncid))
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(time)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')
    call cdfcopy(r(1),val,flip=lflip)
    
    ! write variable
    if (present(time)) then
       call cdferr(nf90_put_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_put_var(cdf_ncid,varid,r(1)))
    end if

    ! put attributes
    if (present(units)) &
         call cdfattput(name,'units',units)
    if (present(long_name)) &
         call cdfattput(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattput(name,'missing_value',missing_value)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr0put


  !**************************************
  ! write a time dependent variable of real kind (1d)
  subroutine cdfvari1put(name,val,dims,flip,time,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    integer, intent(in) :: val(:)
    character(len=*), intent(in) :: dims(:)
    logical, intent(in), optional :: flip
    real (kind=wp), intent(in),optional :: time
    integer, optional :: slice(:,:)
    character(len=*), intent(in),optional :: units
    character(len=*), intent(in), optional :: long_name    
    integer, intent(in),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvari1put'
    integer, allocatable :: r(:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),itmp(nf90_max_dims)
    integer :: rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=1
    if (present(time)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! check space dimensions
    if (nds.gt.0) then
       call cdferror(size(dims).ne.nds, &
            subname//': number of dimension names specified incorrect')
       do i=1,nds
          call cdferr(nf90_inq_dimid(cdf_ncid,dims(i),dimids(i)))
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       call cdfinvertvec(dimids(1:nds),flip=lflip)
       call cdfinvertvec(rlen(1:nds),flip=lflip)
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
            subname//': dimensions in file and variable not consistent')
    end if

    ! time handling
    if (present(time)) then
       call cdftimeput(it,time)
       dimids(nd)=cdf_dtimeid
    end if


    ! check if variable already present
    !   if yes: check dimensions
    !   if no:  define
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,itmp,natts))
       call cdferror(ndims.ne.nd, &
            subname//': variable already in file of different number of dimensions')
       if (nd.gt.0) &
            call cdferror(.not.cdfcheckdim(dimids(1:nd),itmp(1:nd)), &
            subname//': variable already in file of different dimension size')
    else
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_var(cdf_ncid,name,nf90_real,dimids(1:nd),varid))
       call cdferr(nf90_enddef(cdf_ncid))
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time)) then
       lstart(nd)=it
    end if

    ! allocate memory
    allocate(r(lcount(1),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')
    call cdfcopy(r(:,1),val,flip=lflip)

    ! write variable
    if (present(time)) then
       call cdferr(nf90_put_var(cdf_ncid,varid,r(:,1),lstart,lcount,lstride))
    else
       call cdferr(nf90_put_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    end if

    ! put attributes
    if (present(units)) &
         call cdfattput(name,'units',units)
    if (present(long_name)) &
         call cdfattput(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattput(name,'missing_value',missing_value)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvari1put


  !**************************************
  ! write a time dependent variable of real kind (0d)
  subroutine cdfvari0put(name,val,dims,flip,time,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    integer, intent(in) :: val
    character(len=*), intent(in) :: dims(:)
    logical, intent(in), optional :: flip
    real (kind=wp), intent(in), optional :: time
    integer, optional :: slice(:,:)
    character(len=*), intent(in), optional :: units
    character(len=*), intent(in), optional :: long_name    
    integer, intent(in),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvari0put'
    integer, allocatable :: r(:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),itmp(nf90_max_dims)
    integer :: rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=0
    if (present(time)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! check space dimensions
    if (nds.gt.0) then
       call cdferror(size(dims).ne.nds, &
            subname//': number of dimension names specified incorrect')
       do i=1,nds
          call cdferr(nf90_inq_dimid(cdf_ncid,dims(i),dimids(i)))
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
            subname//': space dimensions in file and variable not consistent')
    end if

    ! time handling
    if (present(time)) then
       call cdftimeput(it,time)
       dimids(nd)=cdf_dtimeid
    end if

    ! check if variable already present
    !   if yes: check dimensions
    !   if no:  define
    ierr=nf90_inq_varid(cdf_ncid,name,varid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,itmp,natts))
       call cdferror(ndims.ne.nd, &
            subname//': variable already in file of different number of dimensions')
       if (nd.gt.0) &
            call cdferror(.not.cdfcheckdim(dimids(1:nd),itmp(1:nd)), &
            subname//': variable already in file of different dimension size')
    else
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_var(cdf_ncid,name,nf90_real,dimids(1:nd),varid))
       call cdferr(nf90_enddef(cdf_ncid))
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(time)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')
    call cdfcopy(r(1),val,flip=lflip)
    
    ! write variable
    if (present(time)) then
       call cdferr(nf90_put_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_put_var(cdf_ncid,varid,r(1)))
    end if

    ! put attributes
    if (present(units)) &
         call cdfattput(name,'units',units)
    if (present(long_name)) &
         call cdfattput(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattput(name,'missing_value',missing_value)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvari0put


  !**************************************
  ! read a variable of real kind (3d)
  subroutine cdfvarr3get(name,val,flip,time,tindex,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: flip
    real (kind=wp), optional :: time
    integer, optional :: tindex
    integer, optional :: slice(:,:)

    ! out
    real (kind=wp), intent(out) :: val(:,:,:)
    character(len=*), intent(out), optional :: units
    character(len=*), intent(out), optional :: long_name    
    real (kind=wp), intent(out),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr3get'
    real (kind=wp), allocatable :: r(:,:,:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    real (kind=wp) :: add_offset, scale_factor
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time) .or. present(tindex)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=3
    if (present(time) .or. present(tindex)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! get variable
    call cdferror(.not.cdfvaryes(name), &
         subname//': variable ('//trim(name)//') not present in file')
    call cdferr(nf90_inq_varid(cdf_ncid,name,varid))
    call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,dimids,natts))

    ! check total number of dimensions
    call cdferror(ndims.ne.nd,subname//' for a variable which is not 3d in file')

    ! check space dimensions
    if (nds.gt.0) then
       do i=1,nds
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
         subname//': space dimensions of field in file and variable not consistent')
    end if

    ! time handling
    if (present(time) .or. present(tindex)) then
       call cdferror(present(time) .and. present(tindex), &
            subname//': cannot specify time and tindex at same time')
       if (present(time)) then
          call cdftimiget(it,time)
       else
          it=tindex
       end if
       call cdferror(dimids(nd).ne.cdf_dtimeid, &
            subname//': variable in file does not have fourth dim as time')
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time) .or. present(tindex)) then
       lstart(nd)=it
    end if

    ! allocate memory
    allocate(r(lcount(1),lcount(2),lcount(3),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')

    ! read variable
    if (present(time) .or. present(tindex)) then
       call cdferr(nf90_get_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_get_var(cdf_ncid,varid,r(:,:,:,1),lstart,lcount,lstride))
    end if

    ! get attributes
    if (present(units)) &
         call cdfattget(name,'units',units)
    if (present(long_name)) &
         call cdfattget(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattget(name,'missing_value',missing_value)

    ! post-process data
    add_offset=zero
    scale_factor=one
    if (cdfattyes(name,'add_offset')) &
         call cdfattget(name,'add_offset',add_offset)
    if (cdfattyes(name,'scale_factor')) &
         call cdfattget(name,'scale_factor',scale_factor)
    if (add_offset.ne.zero .or. scale_factor.ne.one) then
       if (present(missing_value)) then
          where (r.ne.missing_value) r=r*scale_factor+add_offset
       else
          r=r*scale_factor+add_offset
       end if
                   
    end if
    call cdfcopy(val,r(:,:,:,1),flip=lflip)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr3get


  !**************************************
  ! read a variable of real kind (2d)
  subroutine cdfvarr2get(name,val,flip,time,tindex,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: flip
    real (kind=wp), optional :: time
    integer, optional :: tindex
    integer, optional :: slice(:,:)

    ! out
    real (kind=wp), intent(out) :: val(:,:)
    character(len=*), intent(out), optional :: units
    character(len=*), intent(out), optional :: long_name    
    real (kind=wp), intent(out),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr2get'
    real (kind=wp), allocatable :: r(:,:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    real (kind=wp) :: add_offset, scale_factor
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time) .or. present(tindex)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=2
    if (present(time) .or. present(tindex)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! get variable
    call cdferror(.not.cdfvaryes(name), &
         subname//': variable ('//trim(name)//') not present in file')
    call cdferr(nf90_inq_varid(cdf_ncid,name,varid))
    call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,dimids,natts))

    ! check total number of dimensions
    call cdferror(ndims.ne.nd,subname//' for a variable which is not 2d in file')

    ! check space dimensions
    if (nds.gt.0) then
       do i=1,nds
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
         subname//': space dimensions of field in file and variable not consistent')
    end if

    ! time handling
    if (present(time) .or. present(tindex)) then
       call cdferror(present(time) .and. present(tindex), &
            subname//': cannot specify time and tindex at same time')
       if (present(time)) then
          call cdftimiget(it,time)
       else
          it=tindex
       end if
       call cdferror(dimids(nd).ne.cdf_dtimeid, &
            subname//': variable in file does not have fourth dim as time')
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time) .or. present(tindex)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(lcount(1),lcount(2),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')

    ! read variable
    if (present(time) .or. present(tindex)) then
       call cdferr(nf90_get_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_get_var(cdf_ncid,varid,r(:,:,1),lstart,lcount,lstride))
    end if

    ! get attributes
    if (present(units)) &
         call cdfattget(name,'units',units)
    if (present(long_name)) &
         call cdfattget(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattget(name,'missing_value',missing_value)

    ! post-process data
    add_offset=zero
    scale_factor=one
    if (cdfattyes(name,'add_offset')) &
         call cdfattget(name,'add_offset',add_offset)
    if (cdfattyes(name,'scale_factor')) &
         call cdfattget(name,'scale_factor',scale_factor)
    if (add_offset.ne.zero .or. scale_factor.ne.one) then
       if (present(missing_value)) then
          where (r.ne.missing_value) r=r*scale_factor+add_offset
       else
          r=r*scale_factor+add_offset
       end if
    end if
    call cdfcopy(val,r(:,:,1),flip=lflip)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr2get


  !**************************************
  ! read a variable of real kind (1d)
  subroutine cdfvarr1get(name,val,flip,time,tindex,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: flip
    real (kind=wp), optional :: time
    integer, optional :: tindex
    integer, optional :: slice(:,:)

    ! out
    real (kind=wp), intent(out) :: val(:)
    character(len=*), intent(out), optional :: units
    character(len=*), intent(out), optional :: long_name    
    real (kind=wp), intent(out),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr1get'
    real (kind=wp), allocatable :: r(:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    real (kind=wp) :: add_offset, scale_factor
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time) .or. present(tindex)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=1
    if (present(time) .or. present(tindex)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! get variable
    call cdferror(.not.cdfvaryes(name), &
         subname//': variable ('//trim(name)//') not present in file')
    call cdferr(nf90_inq_varid(cdf_ncid,name,varid))
    call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,dimids,natts))

    ! check total number of dimensions
    call cdferror(ndims.ne.nd,subname//' for a variable which is not 1d in file')

    ! check space dimensions
    if (nds.gt.0) then
       do i=1,nds
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
         subname//': space dimensions of field in file and variable not consistent')
    end if

    ! time handling
    if (present(time) .or. present(tindex)) then
       call cdferror(present(time) .and. present(tindex), &
            subname//': cannot specify time and tindex at same time')
       if (present(time)) then
          call cdftimiget(it,time)
       else
          it=tindex
       end if
       call cdferror(dimids(nd).ne.cdf_dtimeid, &
            subname//': variable in file does not have fourth dim as time')
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time) .or. present(tindex)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(lcount(1),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')

    ! read variable
    if (present(time) .or. present(tindex)) then
       call cdferr(nf90_get_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_get_var(cdf_ncid,varid,r(:,1),lstart,lcount,lstride))
    end if

    ! get attributes
    if (present(units)) &
         call cdfattget(name,'units',units)
    if (present(long_name)) &
         call cdfattget(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattget(name,'missing_value',missing_value)

    ! post-process data
    add_offset=zero
    scale_factor=one
    if (cdfattyes(name,'add_offset')) &
         call cdfattget(name,'add_offset',add_offset)
    if (cdfattyes(name,'scale_factor')) &
         call cdfattget(name,'scale_factor',scale_factor)
    if (add_offset.ne.zero .or. scale_factor.ne.one) then
       if (present(missing_value)) then
          where (r.ne.missing_value) r=r*scale_factor+add_offset
       else
          r=r*scale_factor+add_offset
       end if
    end if
    call cdfcopy(val,r(:,1),flip=lflip)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr1get


  !**************************************
  ! read a variable of real kind (0d)
  subroutine cdfvarr0get(name,val,flip,time,tindex,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: flip
    real (kind=wp), optional :: time
    integer, optional :: tindex
    integer, optional :: slice(:,:)

    ! out
    real (kind=wp), intent(out) :: val
    character(len=*), intent(out), optional :: units
    character(len=*), intent(out), optional :: long_name    
    real (kind=wp), intent(out),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvarr0get'
    real (kind=wp), allocatable :: r(:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    real (kind=wp) :: add_offset, scale_factor
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time) .or. present(tindex)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=0
    if (present(time) .or. present(tindex)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! get variable
    call cdferror(.not.cdfvaryes(name), &
         subname//': variable ('//trim(name)//') not present in file')
    call cdferr(nf90_inq_varid(cdf_ncid,name,varid))
    call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,dimids,natts))

    ! check total number of dimensions
    call cdferror(ndims.ne.nd,subname//' for a variable which is not 0d in file')

    ! check space dimensions
    if (nds.gt.0) then
       do i=1,nds
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
         subname//': space dimensions of field in file and variable not consistent')
    end if

    ! time handling
    if (present(time) .or. present(tindex)) then
       call cdferror(present(time) .and. present(tindex), &
            subname//': cannot specify time and tindex at same time')
       if (present(time)) then
          call cdftimiget(it,time)
       else
          it=tindex
       end if
       call cdferror(dimids(nd).ne.cdf_dtimeid, &
            subname//': variable in file does not have fourth dim as time')
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(time) .or. present(tindex)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')

    ! read variable
    if (present(time) .or. present(tindex)) then
       call cdferr(nf90_get_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_get_var(cdf_ncid,varid,r(1)))
    end if

    ! get attributes
    if (present(units)) &
         call cdfattget(name,'units',units)
    if (present(long_name)) &
         call cdfattget(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattget(name,'missing_value',missing_value)

    ! post-process data
    add_offset=zero
    scale_factor=one
    if (cdfattyes(name,'add_offset')) &
         call cdfattget(name,'add_offset',add_offset)
    if (cdfattyes(name,'scale_factor')) &
         call cdfattget(name,'scale_factor',scale_factor)
    if (add_offset.ne.zero .or. scale_factor.ne.one) then
       if (present(missing_value)) then
          where (r.ne.missing_value) r=r*scale_factor+add_offset
       else
          r=r*scale_factor+add_offset
       end if
    end if
    call cdfcopy(val,r(1),flip=lflip)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvarr0get


  !**************************************
  ! read a variable of integer kind (1d)
  subroutine cdfvari1get(name,val,flip,time,tindex,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: flip
    real (kind=wp), optional :: time
    integer, optional :: tindex
    integer, optional :: slice(:,:)

    ! out
    integer, intent(out) :: val(:)
    character(len=*), intent(out), optional :: units
    character(len=*), intent(out), optional :: long_name    
    integer, intent(out),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvari1get'
    integer, allocatable :: r(:,:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    integer :: add_offset, scale_factor
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time) .or. present(tindex)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=1
    if (present(time) .or. present(tindex)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! get variable
    call cdferror(.not.cdfvaryes(name), &
         subname//': variable ('//trim(name)//') not present in file')
    call cdferr(nf90_inq_varid(cdf_ncid,name,varid))
    call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,dimids,natts))

    ! check total number of dimensions
    call cdferror(ndims.ne.nd,subname//' for a variable which is not 1d in file')

    ! check space dimensions
    if (nds.gt.0) then
       do i=1,nds
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
         subname//': space dimensions of field in file and variable not consistent')
    end if

    ! time handling
    if (present(time) .or. present(tindex)) then
       call cdferror(present(time) .and. present(tindex), &
            subname//': cannot specify time and tindex at same time')
       if (present(time)) then
          call cdftimiget(it,time)
       else
          it=tindex
       end if
       call cdferror(dimids(nd).ne.cdf_dtimeid, &
            subname//': variable in file does not have fourth dim as time')
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(slice)) then
       lstart(1:nds)=slice(:,1)
       lcount(1:nds)=slice(:,2)-slice(:,1)+1
       call cdfinvertvec(lstart(1:nds),flip=lflip)
       call cdfinvertvec(lcount(1:nds),flip=lflip)
    else
       lcount(1:nds)=rlen(1:nds)
    end if
    if (present(time) .or. present(tindex)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(lcount(1),1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')

    ! read variable
    if (present(time) .or. present(tindex)) then
       call cdferr(nf90_get_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_get_var(cdf_ncid,varid,r(:,1),lstart,lcount,lstride))
    end if

    ! get attributes
    if (present(units)) &
         call cdfattget(name,'units',units)
    if (present(long_name)) &
         call cdfattget(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattget(name,'missing_value',missing_value)

    ! post-process data
    add_offset=zero
    scale_factor=one
    if (cdfattyes(name,'add_offset')) &
         call cdfattget(name,'add_offset',add_offset)
    if (cdfattyes(name,'scale_factor')) &
         call cdfattget(name,'scale_factor',scale_factor)
    if (add_offset.ne.zero .or. scale_factor.ne.one) then
       if (present(missing_value)) then
          where (r.ne.missing_value) r=r*scale_factor+add_offset
       else
          r=r*scale_factor+add_offset
       end if
    end if
    call cdfcopy(val,r(:,1),flip=lflip)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvari1get


  !**************************************
  ! read a variable of integer kind (0d)
  subroutine cdfvari0get(name,val,flip,time,tindex,slice, &
       units,long_name,missing_value)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: flip
    real (kind=wp), optional :: time
    integer, optional :: tindex
    integer, optional :: slice(:,:)

    ! out
    integer, intent(out) :: val
    character(len=*), intent(out), optional :: units
    character(len=*), intent(out), optional :: long_name    
    integer, intent(out),optional :: missing_value

    ! local (dimension dependent)
    character (len=11), parameter :: subname='cdfvari0get'
    integer, allocatable :: r(:)
    ! local (rest)
    character(len=256) :: stmp
    integer :: varid,xtype,ndims,natts
    integer :: dimids(nf90_max_dims),rlen(nf90_max_dims),llen(nf90_max_dims)
    integer :: i,ierr,it,nd,nds
    integer :: lstart(nf90_max_dims),lcount(nf90_max_dims),lstride(nf90_max_dims)
    integer :: add_offset, scale_factor
    logical :: lflip

    call cdfdebug(subname//': entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,subname//' called without active session')
    if (present(time) .or. present(tindex)) &
         call cdferror(.not.cdf_tactive,subname//' called without time defined')

    ! flipping
    if (present(flip)) then
       lflip=.not.flip
    else
       lflip=.true.
    end if

    ! space dimensions
    nds=0
    if (present(time) .or. present(tindex)) then
       nd=nds+1
    else
       nd=nds
    end if

    ! check slicing
    if (present(slice) .and. nds.gt.0) then
       call cdferror(size(slice,1).ne.nds .or. size(slice,2).ne.2 .or. &
            maxval(slice).le.0 .or. minval(slice(:,2)-slice(:,1)).lt.1, &
            subname//': check dimensioning of slice')
    end if

    ! get variable
    call cdferror(.not.cdfvaryes(name), &
         subname//': variable ('//trim(name)//') not present in file')
    call cdferr(nf90_inq_varid(cdf_ncid,name,varid))
    call cdferr(nf90_inquire_variable(cdf_ncid,varid,stmp,xtype,ndims,dimids,natts))

    ! check total number of dimensions
    call cdferror(ndims.ne.nd,subname//' for a variable which is not 0d in file')

    ! check space dimensions
    if (nds.gt.0) then
       do i=1,nds
          call cdferr(nf90_inquire_dimension(cdf_ncid,dimids(i),stmp,rlen(i)))
       end do
       llen(1:nds)=shape(val)
       call cdferror(.not.cdfcheckdim(llen(1:nds),rlen(1:nds),flip=lflip,slice=slice), &
         subname//': space dimensions of field in file and variable not consistent')
    end if

    ! time handling
    if (present(time) .or. present(tindex)) then
       call cdferror(present(time) .and. present(tindex), &
            subname//': cannot specify time and tindex at same time')
       if (present(time)) then
          call cdftimiget(it,time)
       else
          it=tindex
       end if
       call cdferror(dimids(nd).ne.cdf_dtimeid, &
            subname//': variable in file does not have fourth dim as time')
    end if

    ! compute domain
    lstart(:)=1
    lcount(:)=1
    lstride(:)=1
    if (present(time) .or. present(tindex)) then
       lstart(nd)=it
    end if

    ! prepare data
    allocate(r(1),stat=ierr)
    call cdferror(ierr.ne.0,subname//': could not allocate memory storage')

    ! read variable
    if (present(time) .or. present(tindex)) then
       call cdferr(nf90_get_var(cdf_ncid,varid,r,lstart,lcount,lstride))
    else
       call cdferr(nf90_get_var(cdf_ncid,varid,r(1)))
    end if

    ! get attributes
    if (present(units)) &
         call cdfattget(name,'units',units)
    if (present(long_name)) &
         call cdfattget(name,'long_name',long_name)
    if (present(missing_value)) &
         call cdfattget(name,'missing_value',missing_value)

    ! post-process data
    add_offset=zero
    scale_factor=one
    if (cdfattyes(name,'add_offset')) &
         call cdfattget(name,'add_offset',add_offset)
    if (cdfattyes(name,'scale_factor')) &
         call cdfattget(name,'scale_factor',scale_factor)
    if (add_offset.ne.zero .or. scale_factor.ne.one) then
       if (present(missing_value)) then
          where (r.ne.missing_value) r=r*scale_factor+add_offset
       else
          r=r*scale_factor+add_offset
       end if
    end if
    call cdfcopy(val,r(1),flip=lflip)

    ! release memory
    deallocate(r)

    call cdfdebug(subname//': exit')

  end subroutine cdfvari0get


  !**************************************
  ! write an attribute of real kind
  subroutine cdfattrput(var,name,val)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var,name
    real (kind=wp), intent(in) :: val

    ! local
    integer :: varid,ierr

    call cdfdebug('cdfattrput: entry '//trim(var)//' '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfattrput called without active session')

    ! get id of variable to attribute
    if (var.ne.'') then
       call cdferr(nf90_inq_varid(cdf_ncid,var,varid))
    else
       varid=NF90_GLOBAL
    end if

    ! put file in define mode
    call cdferr(nf90_redef(cdf_ncid))

    ! check if attribute exists and delete
    ierr=nf90_inquire_attribute(cdf_ncid,varid,name)
    if (ierr.eq.0) &
         call cdferr(nf90_del_att(cdf_ncid,varid,name))

    ! put attribute
    call cdferr(nf90_put_att(cdf_ncid,varid,name,val))

    ! put file in data mode
    call cdferr(nf90_enddef(cdf_ncid))

    call cdfdebug('cdfattrput: exit')

  end subroutine cdfattrput


  !**************************************
  ! read an attribute of real kind
  subroutine cdfattrget(var,name,val)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var,name

    ! out
    real (kind=wp), intent(out) :: val

    ! local
    integer :: varid

    call cdfdebug('cdfattrget: entry '//trim(var)//' '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfattrget called without active session')

    ! get id of variable to attribute
    if (var.ne.'') then
       call cdferr(nf90_inq_varid(cdf_ncid,var,varid))
    else
       varid=NF90_GLOBAL
    end if

    ! get attribute
    call cdferr(nf90_get_att(cdf_ncid,varid,name,val))

    call cdfdebug('cdfattrget: exit')

  end subroutine cdfattrget


  !**************************************
  ! write an attribute of integer kind
  subroutine cdfattiput(var,name,val)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var,name
    integer, intent(in) :: val

    ! local
    integer :: varid,ierr

    call cdfdebug('cdfattiput: entry '//trim(var)//' '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfattiput called without active session')

    ! get id of variable to attribute
    if (var.ne.'') then
       call cdferr(nf90_inq_varid(cdf_ncid,var,varid))
    else
       varid=NF90_GLOBAL
    end if

    ! put file in define mode
    call cdferr(nf90_redef(cdf_ncid))

    ! check if attribute exists and delete
    ierr=nf90_inquire_attribute(cdf_ncid,varid,name)
    if (ierr.eq.0) &
         call cdferr(nf90_del_att(cdf_ncid,varid,name))

    ! put attribute
    call cdferr(nf90_put_att(cdf_ncid,varid,name,val))

    ! put file in data mode
    call cdferr(nf90_enddef(cdf_ncid))

    call cdfdebug('cdfattiput: exit')

  end subroutine cdfattiput


  !**************************************
  ! read an attribute of integer kind
  subroutine cdfattiget(var,name,val)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var,name

    ! out
    integer, intent(out) :: val

    ! local
    integer :: varid

    call cdfdebug('cdfattiget: entry '//trim(var)//' '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfattiget called without active session')

    ! get id of variable to attribute
    if (var.ne.'') then
       call cdferr(nf90_inq_varid(cdf_ncid,var,varid))
    else
       varid=NF90_GLOBAL
    end if

    ! get attribute
    call cdferr(nf90_get_att(cdf_ncid,varid,name,val))

    call cdfdebug('cdfattiget: exit')

  end subroutine cdfattiget


  !**************************************
  ! write an attribute of character kind
  subroutine cdfatttput(var,name,val)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var,name
    character(len=*), intent(in) :: val

    ! local
    integer :: varid,ierr

    call cdfdebug('cdfatttput: entry '//trim(var)//' '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfattput called without active session')

    ! get id of variable to attribute
    if (var.ne.'') then
       call cdferr(nf90_inq_varid(cdf_ncid,var,varid))
    else
       varid=NF90_GLOBAL
    end if

    ! put file in define mode
    call cdferr(nf90_redef(cdf_ncid))

    ! check if attribute exists and delete
    ierr=nf90_inquire_attribute(cdf_ncid,varid,name)
    if (ierr.eq.0) &
         call cdferr(nf90_del_att(cdf_ncid,varid,name))

    ! put attribute
    call cdferr(nf90_put_att(cdf_ncid,varid,name,val))

    ! put file in data mode
    call cdferr(nf90_enddef(cdf_ncid))

    call cdfdebug('cdfatttput: exit')

  end subroutine cdfatttput


  !**************************************
  ! read an attribute of character kind
  subroutine cdfatttget(var,name,val)
    implicit none
    
    ! in
    character(len=*), intent(in) :: var,name

    ! out
    character(len=*), intent(out) :: val

    ! local
    integer :: varid,llen

    call cdfdebug('cdfatttget: entry'//trim(var)//' '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfattget called without active session')

    ! get id of variable to attribute
    if (var.ne.'') then
       call cdferr(nf90_inq_varid(cdf_ncid,var,varid))
    else
       varid=NF90_GLOBAL
    end if

    ! get attribute
    call cdferr(nf90_inquire_attribute(cdf_ncid,varid,name,len=llen))
    call cdferror(llen+1.gt.len(val),'cdfattget string not sufficiently long')
    val=' '
    call cdferr(nf90_get_att(cdf_ncid,varid,name,val))
    val(llen+1:llen+1)=char(0)

    call cdfdebug('cdfatttget: exit')

  end subroutine cdfatttget


  !**************************************
  ! write a dimension
  subroutine cdfdimput(name,llen)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name
    integer, intent(in) :: llen

    ! local
    character(len=256) :: stmp
    integer :: dimid,ierr,itmp

    call cdfdebug('cdfdimput: entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfdimput called without active session')

    ! check dimension
    call cdferror(llen.lt.1, &
         'cdfdimput: dimensions must be greater or equal to one')

    ! check if dimension already exists
    ierr=nf90_inq_dimid(cdf_ncid,name,dimid)
    if (ierr.eq.0) then
       call cdferr(nf90_inquire_dimension(cdf_ncid,dimid,stmp,itmp))
       call cdferror(itmp.ne.llen, &
            'cdfdimput: dimension exists and is not of same size!')
       return
    end if

    ! put file in define mode
    call cdferr(nf90_redef(cdf_ncid))

    ! define dimension
    call cdferr(nf90_def_dim(cdf_ncid,name,llen,dimid))

    ! put file in data mode
    call cdferr(nf90_enddef(cdf_ncid))

    call cdfdebug('cdfdimput: exit')

  end subroutine cdfdimput


  !**************************************
  ! read a dimension
  subroutine cdfdimget(name,len)
    implicit none
    
    ! in
    character(len=*), intent(in) :: name

    ! out
    integer, intent(out) :: len

    ! local
    character(len=256) :: stmp
    integer :: dimid

    call cdfdebug('cdfdimget: entry '//trim(name))

    ! check if session active
    call cdferror(.not.cdf_active,'cdfdimget called without active session')

    ! get dimension
    call cdferr(nf90_inq_dimid(cdf_ncid,name,dimid))
    call cdferr(nf90_inquire_dimension(cdf_ncid,dimid,stmp,len))

    call cdfdebug('cdfdimget: exit')

  end subroutine cdfdimget


  !**************************************
  ! define time dimension and units
  subroutine cdftime(dtime,dtimeu)
    implicit none

    ! in
    character(len=*), intent(in) :: dtime,dtimeu

    ! local
    integer :: ierr,tvid,dimid(1)
    character(len=256) :: stmp

    call cdfdebug('cdftime: entry '//trim(dtime)//' '//trim(dtimeu))

    ! check if session active
    call cdferror(.not.cdf_active,'cdftime called without active session')
    call cdferror(cdf_tactive,'cdftime called twice for one session')

    ! begin session
    cdf_dtime=dtime
    cdf_dtimeu=dtimeu
    cdf_dtimeid=0
    cdf_tactive=.true.

    ! check if time dimension is already present
    ierr=nf90_inq_dimid(cdf_ncid,cdf_dtime,cdf_dtimeid)
    if (ierr.ne.0) then
       ! define time dimension
       call cdferr(nf90_redef(cdf_ncid))
       call cdferr(nf90_def_dim(cdf_ncid,cdf_dtime,NF90_UNLIMITED,cdf_dtimeid))
       ! define time variable
       dimid(1)=cdf_dtimeid
       call cdferr(nf90_def_var(cdf_ncid,cdf_dtime,nf90_real,dimid,tvid))
       call cdferr(nf90_enddef(cdf_ncid))
       call cdfatttput(cdf_dtime,'units',cdf_dtimeu)
    else
       call cdferr(nf90_inq_varid(cdf_ncid,cdf_dtime,tvid))
       call cdfatttget(cdf_dtime,'units',stmp)
       call cdferror(cdf_dtimeu(1:len_trim(cdf_dtimeu)).ne.stmp(1:len_trim(cdf_dtimeu)), &
            'cdftime: units of time variable is not correct in file')
    end if

    call cdfdebug('cdftime: exit')

  end subroutine cdftime


  !**************************************
  ! begin cdf session and store details
  subroutine cdfbegin(fname,caller,create,overwrite,readonly)
    implicit none

    ! in
    character(len=*), intent(in) :: fname,caller
    logical, intent(in), optional :: create,overwrite,readonly

    ! local
    integer :: ncid,ncmode

    call cdfdebug('cdfbegin: entry '//trim(fname)//' '//trim(caller))

    ! check if session active
    call cdferror(cdf_active,'cdfbegin called with active session')

    ! begin session
    cdf_caller=caller
    cdf_fname=fname

    ! create file (if necessary)
    if (present(create)) then
       if (create) then
          ncmode=NF90_NOCLOBBER
          if (present(overwrite)) then
             if (overwrite) then
                ncmode=NF90_CLOBBER
             end if
          end if
          call cdferr(nf90_create(fname,ncmode,ncid))
          call cdferr(nf90_enddef(ncid,100*1024)) ! add some header space
          call cdferr(nf90_close(ncid))
       end if
    end if
    
    ! open file
    ncmode=NF90_WRITE
    if (present(readonly)) then
       if (readonly) then
          ncmode=NF90_NOWRITE
       end if
    end if
    call cdferr(nf90_open(fname,ncmode,ncid))

    ! set globals
    cdf_active=.true.
    cdf_ncid=ncid
    cdf_tactive=.false.
    cdf_dtime=' '
    cdf_dtimeu=' '
    cdf_dtimeid=0

    call cdfdebug('cdfbegin: exit')

  end subroutine cdfbegin


  !**************************************
  ! end cdf session
  subroutine cdfend()
    implicit none

    call cdfdebug('cdfend: entry')

    ! check if session active
    call cdferror(.not.cdf_active,'cdfend called without active session')

    ! close session

    call cdferr(nf90_close(cdf_ncid))
    cdf_active=.false.
    cdf_fname=''
    cdf_caller=''
    cdf_tactive=.false.
    cdf_dtime=''
    cdf_dtimeu=''
    cdf_dtimeid=0
    cdf_ncid=0

    call cdfdebug('cdfend: exit')

  end subroutine cdfend


  !**************************************
  ! get time index
  subroutine cdftimiget(it,t)
    implicit none

    ! in
    real (kind=wp), intent(in) :: t

    ! out
    integer, intent(out) :: it

    ! local
    integer :: nt,tvid,xtype,ndims,dimids(9),natts,ierr
    character(len=256) :: stmp
    real (kind=wp), allocatable :: ts(:)

    call cdfdebug('cdftimiget: entry')

    ! check if session active
    call cdferror(.not.cdf_active .or. .not.cdf_tactive, &
         'cdftimiget called without active session')

    ! get time variable
    call cdferr(nf90_inq_varid(cdf_ncid,cdf_dtime,tvid))
    call cdferr(nf90_inquire_variable(cdf_ncid,tvid,stmp,xtype,ndims,dimids,natts))
    call cdferror(ndims.ne.1,'cdftimiget: time variable not 1d')
    call cdferror(dimids(1).ne.cdf_dtimeid,'cdftimiget: inconsistent time dimension')

    ! get time dimension
    call cdferr(nf90_inquire_dimension(cdf_ncid,cdf_dtimeid,stmp,nt))

    ! get time array
    allocate(ts(nt),stat=ierr)
    call cdferror(ierr.ne.0,'cdftimiget: could not allocate memory storage')
    call cdferr(nf90_get_var(cdf_ncid,tvid,ts))
    do it=1,nt
       if (ts(it).eq.t) exit
    end do
    call cdferror(it.gt.nt,'cdftimiget: time not found')
    deallocate(ts)

    call cdfdebug('cdftimiget: exit')

  end subroutine cdftimiget


  !**************************************
  ! get time corresponding to index
  subroutine cdftimeget(it,t)
    implicit none

    ! in
    integer, intent(in) :: it

    ! out
    real (kind=wp), intent(out) :: t

    ! local
    integer :: nt,tvid,xtype,ndims,dimids(9),natts,ierr
    character(len=256) :: stmp
    real (kind=wp), allocatable :: ts(:)

    call cdfdebug('cdftimeget: entry')

    ! check if session active
    call cdferror(.not.cdf_active .or. .not.cdf_tactive, &
         'cdftimeget called without active session')

    ! get time variable
    call cdferr(nf90_inq_varid(cdf_ncid,cdf_dtime,tvid))
    call cdferr(nf90_inquire_variable(cdf_ncid,tvid,stmp,xtype,ndims,dimids,natts))
    call cdferror(ndims.ne.1,'cdftimeget: time variable not 1d')
    call cdferror(dimids(1).ne.cdf_dtimeid,'cdftimeget: inconsistent time dimension')

    ! get time dimension
    call cdferr(nf90_inquire_dimension(cdf_ncid,cdf_dtimeid,stmp,nt))

    ! get time array
    allocate(ts(nt),stat=ierr)
    call cdferror(ierr.ne.0,'cdftimeget: could not allocate memory storage')
    call cdferr(nf90_get_var(cdf_ncid,tvid,ts))
    
    ! return result
    call cdferror(it.gt.nt,'cdftimeget: time not found')
    t=ts(it)
    deallocate(ts)

    call cdfdebug('cdftimeget: exit')

  end subroutine cdftimeget


  !**************************************
  ! add a time to a NetCDF file
  subroutine cdftimeput(it,t)
    implicit none

    ! in
    real (kind=wp), intent(in) :: t
 
    ! out
    integer, intent(out) :: it

    ! local
    integer :: nt,tvid,ndims,dimids(9),natts,xtype,ierr
    character(len=256) :: stmp
    real (kind=wp), allocatable :: ts(:)

    call cdfdebug('cdftimeput: entry')

    ! check if session active
    call cdferror(.not.cdf_active .or. .not.cdf_tactive, &
         'cdftimeput called without active session')

    ! get time dimension
    call cdferr(nf90_inquire_dimension(cdf_ncid,cdf_dtimeid,stmp,nt))
    
    ! get time variable
    call cdferr(nf90_inq_varid(cdf_ncid,cdf_dtime,tvid))
    call cdferr(nf90_inquire_variable(cdf_ncid,tvid,stmp,xtype,ndims,dimids,natts))
    call cdferror(ndims.ne.1,'cdftimeput: time variable not 1d')
    call cdferror(dimids(1).ne.cdf_dtimeid,'cdftimeput: inconsistent time dimension')
    if (nt.gt.0) then
       ! after first time
       allocate(ts(nt+1),stat=ierr)
       call cdferror(ierr.ne.0,'cdftimeput: could not allocate memory storage')
       call cdferr(nf90_get_var(cdf_ncid,tvid,ts(1:nt)))
       do it=1,nt
          if (abs(ts(it)-t).lt.0.0001_wp) exit
       end do
       if (it.gt.nt) then
          call cdferror(ts(nt)>t,'cdftimeput: times must be added in increasing order')
          ts(nt+1)=t
          call cdferr(nf90_put_var(cdf_ncid,tvid,ts))
       end if
       deallocate(ts)
    else
       ! first time
       it=1
       allocate(ts(1),stat=ierr)
       call cdferror(ierr.ne.0,'cdftimeput: could not allocate memory storage')
       ts(1)=t
       call cdferr(nf90_put_var(cdf_ncid,tvid,ts))
    end if

    call cdfdebug('cdftimeput: exit')

  end subroutine cdftimeput


  !**************************************
  ! copy and flip 3d field (if necessary)
  subroutine cdfcopyr3(dst,src,flip)
    implicit none

    ! in
    real (kind=wp), intent(in) :: src(:,:,:)
    logical, intent(in), optional :: flip

    ! out
    real (kind=wp), intent(out) :: dst(:,:,:)

    ! local
    integer, parameter :: nd=3
    integer :: i,j,k
    integer :: slb(nd),sub(nd),dlb(nd),dub(nd)

    call cdfdebug('cdfcopyr3: entry')

    ! get dimensions
    slb=lbound(src)
    sub=ubound(src)
    dlb=lbound(dst)
    dub=ubound(dst)

    ! check dimensions
    call cdferror(.not.cdfcheckdim(sub-slb,dub-dlb,flip=flip), &
         'cdfcopyr3: dimension mismatch')

    ! copy and flip it (if necessary)
    if (present(flip)) then
       if (flip) then
          do i=slb(1),sub(1)
             do j=slb(2),sub(2)
                do k=slb(3),sub(3)
                   dst(k-slb(3)+dlb(3),j-slb(2)+dlb(2),i-slb(1)+dlb(1))= &
                        src(i,j,k)
                end do
             end do
          end do
       else
          dst=src
       end if
    else
       dst=src
    end if

    call cdfdebug('cdfcopyr3: exit')

  end subroutine cdfcopyr3


  !**************************************
  ! copy and flip 2d field (if necessary)
  subroutine cdfcopyr2(dst,src,flip)
    implicit none

    ! in
    real (kind=wp), intent(in) :: src(:,:)
    logical, intent(in), optional :: flip

    ! out
    real (kind=wp), intent(out) :: dst(:,:)

    ! local
    integer, parameter :: nd=2
    integer :: i,j
    integer :: slb(nd),sub(nd),dlb(nd),dub(nd)

    call cdfdebug('cdfcopyr2: entry')

    ! get dimensions
    slb=lbound(src)
    sub=ubound(src)
    dlb=lbound(dst)
    dub=ubound(dst)

    ! check dimensions
    call cdferror(.not.cdfcheckdim(sub-slb,dub-dlb,flip=flip), &
         'cdfcopyr2: dimension mismatch')

    ! copy and flip it (if necessary)
    if (present(flip)) then
       if (flip) then
          do i=slb(1),sub(1)
             do j=slb(2),sub(2)
                dst(j-slb(2)+dlb(2),i-slb(1)+dlb(1))= &
                     src(i,j)
             end do
          end do
       else
          dst=src
       end if
    else
       dst=src
    end if

    call cdfdebug('cdfcopyr2: exit')

  end subroutine cdfcopyr2


  !**************************************
  ! copy and flip 1d field (dummy routine, copy only)
  subroutine cdfcopyr1(dst,src,flip)
    implicit none

    ! in
    real (kind=wp), intent(in) :: src(:)
    logical, intent(in), optional :: flip

    ! out
    real (kind=wp), intent(out) :: dst(:)

    ! local
    integer, parameter :: nd=1
    integer :: slb(nd),sub(nd),dlb(nd),dub(nd)

    call cdfdebug('cdfcopyr1: entry')

    ! get dimensions
    slb=lbound(src)
    sub=ubound(src)
    dlb=lbound(dst)
    dub=ubound(dst)

    ! check dimensions
    call cdferror(.not.cdfcheckdim(sub-slb,dub-dlb,flip=flip), &
         'cdfcopyr1: dimension mismatch')

    ! copy
    dst=src

    call cdfdebug('cdfcopyr1: exit')

  end subroutine cdfcopyr1


  !**************************************
  ! copy and flip 0d field (dummy routine, copy only)
  subroutine cdfcopyr0(dst,src,flip)
    implicit none

    ! in
    real (kind=wp), intent(in) :: src
    logical, intent(in), optional :: flip

    ! out
    real (kind=wp), intent(out) :: dst

    call cdfdebug('cdfcopyr0: entry')

    ! copy
    dst=src

    call cdfdebug('cdfcopyr0: exit')

  end subroutine cdfcopyr0


  !**************************************
  ! copy and flip 3d field (if necessary)
  subroutine cdfcopyi3(dst,src,flip)
    implicit none

    ! in
    integer, intent(in) :: src(:,:,:)
    logical, intent(in), optional :: flip

    ! out
    integer, intent(out) :: dst(:,:,:)

    ! local
    integer, parameter :: nd=3
    integer :: i,j,k
    integer :: slb(nd),sub(nd),dlb(nd),dub(nd)

    call cdfdebug('cdfcopyi3: entry')

    ! get dimensions
    slb=lbound(src)
    sub=ubound(src)
    dlb=lbound(dst)
    dub=ubound(dst)

    ! check dimensions
    call cdferror(.not.cdfcheckdim(sub-slb,dub-dlb,flip=flip), &
         'cdfcopyi3: dimension mismatch')

    ! copy and flip it (if necessary)
    if (present(flip)) then
       if (flip) then
          do i=slb(1),sub(1)
             do j=slb(2),sub(2)
                do k=slb(3),sub(3)
                   dst(k-slb(3)+dlb(3),j-slb(2)+dlb(2),i-slb(1)+dlb(1))= &
                        src(i,j,k)
                end do
             end do
          end do
       else
          dst=src
       end if
    else
       dst=src
    end if

    call cdfdebug('cdfcopyi3: exit')

  end subroutine cdfcopyi3


  !**************************************
  ! copy and flip 2d field (if necessary)
  subroutine cdfcopyi2(dst,src,flip)
    implicit none

    ! in
    integer, intent(in) :: src(:,:)
    logical, intent(in), optional :: flip

    ! out
    integer, intent(out) :: dst(:,:)

    ! local
    integer, parameter :: nd=2
    integer :: i,j
    integer :: slb(nd),sub(nd),dlb(nd),dub(nd)

    call cdfdebug('cdfcopyi2: entry')

    ! get dimensions
    slb=lbound(src)
    sub=ubound(src)
    dlb=lbound(dst)
    dub=ubound(dst)

    ! check dimensions
    call cdferror(.not.cdfcheckdim(sub-slb,dub-dlb,flip=flip), &
         'cdfcopyi2: dimension mismatch')

    ! copy and flip it (if necessary)
    if (present(flip)) then
       if (flip) then
          do i=slb(1),sub(1)
             do j=slb(2),sub(2)
                dst(j-slb(2)+dlb(2),i-slb(1)+dlb(1))= &
                     src(i,j)
             end do
          end do
       else
          dst=src
       end if
    else
       dst=src
    end if

    call cdfdebug('cdfcopyi2: exit')

  end subroutine cdfcopyi2


  !**************************************
  ! copy and flip 1d field (dummy routine, copy only)
  subroutine cdfcopyi1(dst,src,flip)
    implicit none

    ! in
    integer, intent(in) :: src(:)
    logical, intent(in), optional :: flip

    ! out
    integer, intent(out) :: dst(:)

    ! local
    integer, parameter :: nd=1
    integer :: slb(nd),sub(nd),dlb(nd),dub(nd)

    call cdfdebug('cdfcopyi1: entry')

    ! get dimensions
    slb=lbound(src)
    sub=ubound(src)
    dlb=lbound(dst)
    dub=ubound(dst)

    ! check dimensions
    call cdferror(.not.cdfcheckdim(sub-slb,dub-dlb,flip=flip), &
         'cdfcopyi1: dimension mismatch')

    ! copy
    dst=src

    call cdfdebug('cdfcopyi1: exit')

  end subroutine cdfcopyi1


  !**************************************
  ! copy and flip 0d field (dummy routine, copy only)
  subroutine cdfcopyi0(dst,src,flip)
    implicit none

    ! in
    integer, intent(in) :: src
    logical, intent(in), optional :: flip

    ! out
    integer, intent(out) :: dst

    call cdfdebug('cdfcopyi0: entry')

    ! copy
    dst=src

    call cdfdebug('cdfcopyi0: exit')

  end subroutine cdfcopyi0


  !**************************************
  ! inverse order of vector elements
  subroutine cdfinvertvec(d,flip)
    implicit none
    
    ! in
    integer, intent(inout) :: d(:)
    logical, intent(in), optional :: flip

    ! local
    integer :: ierr,i,ni
    integer, allocatable :: itmp(:)

    call cdfdebug('cdfinvertvec: entry')
    
    if (present(flip)) then
       if (flip) then
          ni=size(d)
          allocate(itmp(ni),stat=ierr)
          call cdferror(ierr.ne.0, &
               'cdfomvertvec: could not allocate memory storage')
          do i=1,ni
             itmp(i)=d(ni-i+1)
          end do
          d=itmp
          deallocate(itmp)
       end if
    end if

    call cdfdebug('cdfinvertvec: exit')

  end subroutine cdfinvertvec


  !**************************************
  ! compare local and file dimensions
  function cdfcheckdim(d1,d2,flip,slice)
    implicit none

    ! in
    integer, intent(in) :: d1(:)
    integer, intent(in) :: d2(:)
    logical, intent(in), optional :: flip
    integer, intent(in), optional :: slice(:,:)

    ! out
    logical :: cdfcheckdim

    ! local
    integer :: i,nd,ierr
    integer, allocatable :: d3(:)

    call cdfdebug('cdfcheckdim: entry')

    ! default result is false
    cdfcheckdim=.false.

    ! check dimensions
    nd=size(d1)
    if (nd.ne.size(d2)) return
    if (present(slice)) then
         if (nd.ne.size(slice,1) .or. 2.ne.size(slice,2)) return
    end if

    ! copy and flip 
    allocate(d3(nd),stat=ierr)
    call cdferror(ierr.ne.0,'cdfcheckdim: could not allocate memory')
    d3=d2
    call cdfinvertvec(d3,flip=flip)

    ! check dimensioning
    do i=1,nd
       if (present(slice)) then
          ! sliced version
          if (d1(i).gt.d3(i)) return
          if (d1(i).ne.(slice(i,2)-slice(i,1)+1)) return
          if (slice(i,1).gt.d3(i) .or. slice(i,1).lt.1) return
          if (slice(i,2).gt.d3(i) .or. slice(i,2).lt.1) return
          if (slice(i,1).gt.slice(i,2)) return
       else
          ! unsliced version
          if (d1(i).ne.d3(i)) return
       end if
    end do

    ! release memory
    deallocate(d3)

    ! all tests passed, ok!
    cdfcheckdim=.true.

    call cdfdebug('cdfcheckdim: exit')

  end function cdfcheckdim


  !**************************************
  ! check for NetCDF library errors
  subroutine cdferr(status)
    implicit none

    ! in
    integer, intent(in) :: status

    ! local
    character(len=256) message
    integer :: ierr
    
    ! error message to user
    if (status.ne.0) then
       write(0,*) 'CDF MODULE FATAL NETCDF ERROR!'
       write(0,*) 'Error = ',status
       message = nf90_strerror(status)
       write(0,*) 'Message: ',trim(message)
       write(0,*) 'Subroutine: ',trim(cdf_caller)
       write(0,*) 'Execution aborted...'
       if (cdf_active) &
            ierr=nf90_close(cdf_ncid)
       stop 
    end if

  end subroutine cdferr


  ! *************************************************************
  ! echo debugging information (if on)
  subroutine cdfdebug(str)
    implicit none
    
    ! in
    character(len=*) :: str

    if (.not.cdf_debug) return
    write(stdout,*) 'm_cdf> ',trim(str)

  end subroutine cdfdebug


  !**************************************
  ! write error message and terminate
  subroutine cdferror(yes,msg)
    implicit none
    
    ! in
    logical, intent(in) :: yes
    character(len=*) msg
    
    ! local
    integer :: ierr
    
    ! error message to user
    if (yes) then
       write(0,*) 'CDF MODULE FATAL ERROR!'
       write(0,*) 'Message: ',trim(msg)
       write(0,*) 'Subroutine ',trim(cdf_caller)
       write(0,*) 'Execution aborted...'
       if (cdf_active) &
            ierr=nf90_close(cdf_ncid)
       stop
    end if
    
  end subroutine cdferror

#endif
end module m_cdf
