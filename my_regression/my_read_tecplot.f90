program read_tecplot


  implicit none
  
  integer :: ng,ni,nj,nk,nr,num,bs,stream,n, nl, i,j,k
  character(len=1000) :: case, plane
  character(len=1000) :: ch,fgrd,fsol
  double precision :: grd(286,1,301,3), sol(286,1,301,10)
  case="h0.008"
  plane="plate"
  ch="2"
  nr=10
  !integer :: bs,stream,n

  
!  write(ch,*) num
  fgrd='./'//trim(case)//'/'//trim(plane)//'_'//trim(adjustl(ch))//'.x'
  fsol=trim(case)//'/'//trim(plane)//'_'//trim(adjustl(ch))//'.q'
  
  bs=0
  stream=0
  
  ! Choose little-endian or big-endian byte order
  IF (bs .EQ. 1) THEN
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='READ',CONVERT='BIG_ENDIAN',ACCESS='STREAM')
     ELSE
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='READ',CONVERT='BIG_ENDIAN')
     ENDIF !stream
  ELSE
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='READ',CONVERT='LITTLE_ENDIAN',ACCESS='STREAM')
     ELSE
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='READ',CONVERT='LITTLE_ENDIAN')
     ENDIF !stream
  ENDIF !bs
  
  read(10) ng
  PRINT *, ng
  DO n = 1, ng
     read(10) ni, nj, nk
  ENDDO
  PRINT *, ni, nj, nk
  DO n=1,ng
     read(10) grd(1:ni,1:nj,1:nk,1:3)
  ENDDO
  close(10)
  PRINT *, grd(1,1,1,2)

  open(unit=10, file='./'//trim(case)//'/'//trim(plane)//'_'//trim(adjustl(ch))//'.csv', status='unknown')
  DO k = 1, nk
    DO j = 1, nj
      DO i = 1, ni
        write(10, *) grd(i,j,k,1), ',', grd(i,j,k,2), ',', grd(i,j,k,3)
      ENDDO
   ENDDO
 ENDDO
  close(10)

  ! Choose little-endian or big-endian byte order
  IF (bs .EQ. 1) THEN
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='READ',CONVERT='BIG_ENDIAN',ACCESS='STREAM')
     ELSE
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='READ',CONVERT='BIG_ENDIAN')
     ENDIF !stream
  ELSE
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='READ',CONVERT='LITTLE_ENDIAN',ACCESS='STREAM')
     ELSE
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='READ',CONVERT='LITTLE_ENDIAN')
     ENDIF !stream
  ENDIF !bs
  
  read(10) ng
  PRINT *, ng
  DO n = 1, ng
     read(10) ni, nj, nk, nr
  ENDDO
  PRINT *, ni, nj, nk
  DO n=1,ng
     read(10) sol(1:ni,1:nj,1:nk,1:nr)
  ENDDO
  close(10)

  open(unit=10, file='./'//trim(case)//'/'//trim(plane)//'_'//trim(adjustl(ch))//'.csv', status='unknown')
  DO k = 1, nk
    DO j = 1, nj
      DO i = 1, ni
        write(10, *) sol(i,j,k,1)
      ENDDO
   ENDDO
 ENDDO
  close(10)



end program read_tecplot
