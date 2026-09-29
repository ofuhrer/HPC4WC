! ICON
!
! ---------------------------------------------------------------
! Copyright (C) 2004-2024, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
! Contact information: icon-model.org
!
! See AUTHORS.TXT for a list of authors
! See LICENSES/ for license information
! SPDX-License-Identifier: BSD-3-Clause
! ---------------------------------------------------------------

! @brief Master configuration.

MODULE mo_master_config

  USE mtime,       ONLY: MAX_CALENDAR_STR_LEN, MAX_DATETIME_STR_LEN, &
    &                    MAX_TIMEDELTA_STR_LEN
  USE mo_io_units, ONLY: filename_max
  
  IMPLICIT NONE
  
  PUBLIC

  PUBLIC :: calendar_str
  PUBLIC :: addModel, noOfModels, maxNoOfModels
  PUBLIC :: setInstitution
  PUBLIC :: setModelBaseDir, getModelBaseDir
  PUBLIC :: master_component_models
  PUBLIC :: lrestart_write_last
  PUBLIC :: experimentReferenceDate, experimentStartDate, experimentStopDate
  PUBLIC :: checkpointTimeIntval, restartTimeIntval
  PUBLIC :: setRestart, setRestartWriteLast, setReadRestartNamelists
  PUBLIC :: isRestart
  PUBLIC :: read_restart_namelists  
  ! component model configuration
  !_______________________________________________________________________________________________
  !
  !> Holds name, type, and position of processes in the global communicator
  !  of a component model
  
  TYPE t_master_component_model_config
    CHARACTER(len=132)          :: model_name
    CHARACTER(len=filename_max) :: model_namelist_filename
    INTEGER :: model_type
    INTEGER :: model_min_rank
    INTEGER :: model_max_rank
    INTEGER :: model_inc_rank
    INTEGER :: model_rank_group_size
  END TYPE t_master_component_model_config

  INTEGER, PARAMETER :: maxNoOfModels = 16
  INTEGER :: no_of_models = 0
  TYPE(t_master_component_model_config) :: master_component_models(maxNoOfModels)

  ! defaults to DWD to be on the safe side for operational needs  
  CHARACTER(len=256), PROTECTED :: institution = '' 
  
 
  ! restart flag, required for consistent handling in all component models
  
  LOGICAL, PROTECTED :: lrestart = .false.


   LOGICAL, PROTECTED :: read_restart_namelists = .true.
  !> Flag: True, if model run should create restart at experiment end.
  !  This is independent from the settings of the restart interval.
  LOGICAL, PROTECTED ::  lrestart_write_last = .TRUE.

  CHARACTER(len=filename_max), PROTECTED :: model_base_dir = ''

  !> namelist parameters (as raw character strings):

  CHARACTER(len=MAX_CALENDAR_STR_LEN)  :: calendar_str             = ''
  CHARACTER(len=MAX_DATETIME_STR_LEN)  :: experimentReferenceDate  = ''
  CHARACTER(len=MAX_DATETIME_STR_LEN)  :: experimentStartDate      = ''
  CHARACTER(len=MAX_DATETIME_STR_LEN)  :: experimentStopDate       = ''
  CHARACTER(len=MAX_TIMEDELTA_STR_LEN) :: checkpointTimeIntval     = ''
  CHARACTER(len=MAX_TIMEDELTA_STR_LEN) :: restartTimeIntval        = ''
 
  ! ------------------------------------------------------------------------
  ! keep the master memory here to avoid cycle refrences
  ! it is filled by mo_master_init
  ! ------------------------------------------------------------------------
  INTEGER, PARAMETER :: atmo_process         = 1
  INTEGER, PARAMETER :: ocean_process        = 2
  INTEGER, PARAMETER :: ps_radiation_process = 3
  INTEGER, PARAMETER :: hamocc_process       = 4
  INTEGER, PARAMETER :: jsbach_process       = 5
  INTEGER, PARAMETER :: wave_process         = 98
  INTEGER, PARAMETER :: testbed_process      = 99
  INTEGER, PARAMETER :: icon_output_process  = 100
  ! ------------------------------------------------------------------------

  INTEGER :: my_process_model ! =atmo_process,ocean_process,...
  INTEGER :: my_model_no ! 1,2,3  (id uniquely this process, even if it has the
                         ! same my_process_model with other compnents
                         ! Example: Two different components may run the dummy_process
  CHARACTER(len=filename_max) :: my_namelist_filename
  CHARACTER(len=64) :: my_model_name = ""

  CHARACTER(len=filename_max) :: master_namelist_filename = ""

  INTEGER:: my_model_min_rank, my_model_max_rank, my_model_inc_rank 
  LOGICAL :: multiple_models
 
CONTAINS

  SUBROUTINE setInstitution(update_institute)
    CHARACTER(len=*) :: update_institute
    institution = ''  ! needed?
    institution = update_institute
  END SUBROUTINE setInstitution
  
  SUBROUTINE setRestart(lr)
    LOGICAL, INTENT(in) :: lr
    lrestart = lr
  END SUBROUTINE setRestart

  SUBROUTINE setReadRestartNamelists(lr)
    LOGICAL, INTENT(in) :: lr
    read_restart_namelists = lr
  END SUBROUTINE setReadRestartNamelists


  SUBROUTINE setRestartWriteLast(lr)
    LOGICAL, INTENT(in) :: lr
    lrestart_write_last = lr
  END SUBROUTINE setRestartWriteLast

  LOGICAL FUNCTION isRestart()
    isRestart = lrestart
  END FUNCTION isRestart

  SUBROUTINE setModelBaseDir(mbd)
    CHARACTER(len=*), INTENT(in) :: mbd
    model_base_dir = mbd
  END SUBROUTINE setModelBaseDir
  
  CHARACTER(len=filename_max) FUNCTION getModelBaseDir()
    getModelBaseDir = model_base_dir 
  END FUNCTION getModelBaseDir

  SUBROUTINE addModel()
    no_of_models = no_of_models+1
  END SUBROUTINE addModel

  INTEGER FUNCTION noOfModels()
    noOfModels = no_of_models
  END FUNCTION noOfModels
  
END MODULE mo_master_config
