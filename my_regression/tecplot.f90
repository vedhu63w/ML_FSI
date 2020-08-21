subroutine wtec(ng,ni,nj,nk,grd,sol,nr,num,case,plane)

  implicit none
  
  integer, intent(in) :: ng,ni,nj,nk,nr,num
  double precision, intent(in) :: grd(ni,nj,nk,3), sol(ni,nj,nk,nr)
  character(len=1000), intent(in) :: case, plane

  integer :: bs,stream,n
  character(len=1000) :: ch,fgrd,fsol

  
  write(ch,*) num
  fgrd=trim(case)//'/'//trim(plane)//'_'//trim(adjustl(ch))//'.x'
  fsol=trim(case)//'/'//trim(plane)//'_'//trim(adjustl(ch))//'.q'
  
  bs=0
  stream=0
  
  ! Choose little-endian or big-endian byte order
  IF (bs .EQ. 1) THEN
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='BIG_ENDIAN',ACCESS='STREAM',STATUS='REPLACE')
     ELSE
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='BIG_ENDIAN',STATUS='REPLACE')
     ENDIF !stream
  ELSE
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='LITTLE_ENDIAN',ACCESS='STREAM',STATUS='REPLACE')
     ELSE
        OPEN(10,FILE=TRIM(fgrd),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='LITTLE_ENDIAN',STATUS='REPLACE')
     ENDIF !stream
  ENDIF !bs
  
  write(10) ng
  DO n = 1, ng
     write(10) ni, nj, nk
  ENDDO
  DO n=1,ng
     write(10) grd(1:ni,1:nj,1:nk,1:3)
  ENDDO
  close(10)


  ! Choose little-endian or big-endian byte order
  IF (bs .EQ. 1) THEN
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='BIG_ENDIAN',ACCESS='STREAM',STATUS='REPLACE')
     ELSE
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='BIG_ENDIAN',STATUS='REPLACE')
     ENDIF !stream
  ELSE
     ! Switch between streaming I/O
     IF (stream .EQ. 1) THEN
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='LITTLE_ENDIAN',ACCESS='STREAM',STATUS='REPLACE')
     ELSE
        OPEN(10,FILE=TRIM(fsol),FORM='UNFORMATTED',ACTION='WRITE',CONVERT='LITTLE_ENDIAN',STATUS='REPLACE')
     ENDIF !stream
  ENDIF !bs
  
  write(10) ng
  DO n = 1, ng
     write(10) ni, nj, nk, nr
  ENDDO
  DO n=1,ng
     write(10) sol(1:ni,1:nj,1:nk,1:nr)
  ENDDO
  close(10)
  
end subroutine wtec
