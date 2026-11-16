section .text
global asm_kernel_multiply_32_block

; rdi = pointer to A[A_start_row][A_start_col]
; rsi = pointer to B[A_start_col][B_start_col]
; rdx = pointer to C[A_start_row][B_start_col]
; rcx = A_cols = B_rows
; r8 =  B_cols = C_cols

asm_kernel_multiply_32_block:
  push r12
  mov r12, 8
  shl rcx, 3 ; A row stride (sizeof(double) * A_cols)
  shl r8, 3 ; B/C row stride (sizeof(double) * B_cols)
  
OUTER_LOOP:
  xor r11, r11 ;middle loop counter 0 -> 31
  mov r10, rdi ;load A[A_start_row][A_start_col] into r10
MIDDLE_LOOP:
  vmovupd ymm0, [rdx]  ; C[i][j ... j+3]
  mov r9, 4 ;inner loop counter - each loop does 8 FMAs for 32 total
INNER_LOOP:
  vxorpd ymm15, ymm15, ymm15 ; additional accumulator = 0
  vbroadcastsd ymm1, qword [r10] ; A[i][k]
  vbroadcastsd ymm2, qword [r10 + 8]; A[i][k+1]
  vbroadcastsd ymm3, qword [r10 + 16]; A[i][k+2]
  vbroadcastsd ymm4, qword [r10 + 24]; A[i][k+3]
  vbroadcastsd ymm5, qword [r10 + 32]; A[i][k+4]
  vbroadcastsd ymm6, qword [r10 + 40]; A[i][k+5]
  vbroadcastsd ymm7, qword [r10 + 48]; A[i][k+6]
  vmovupd ymm8, [rsi] ; B[k][j ... j+3]
  vmovupd ymm9, [rsi + r8] ; B[k + 1][j ... j+3]
  vmovupd ymm10, [rsi + r8 * 2] ; B[k+ 2][j ... j+3]
  lea rax, [rsi + r8 * 2]
  add rax, r8
  vmovupd ymm11, [rax] ; B[k + 3][j ... j+3]
  vmovupd ymm12, [rsi + r8 * 4] ; B[k + 4][j ... j+3]
  lea rax, [rsi + r8 * 4]
  add rax, r8
  vmovupd ymm13, [rax] ; B[k + 5][j ... j+3]
  lea rax, [rsi + r8 * 4]
  lea rax, [rax + r8 * 2]
  vmovupd ymm14, [rax] ; B[k + 6][j ... j+3]
  vfmadd231pd ymm0, ymm1, ymm8
  vfmadd231pd ymm15, ymm2, ymm9 
  vfmadd231pd ymm0, ymm3, ymm10 
  vfmadd231pd ymm15, ymm4, ymm11 
  vfmadd231pd ymm0, ymm5, ymm12 
  vfmadd231pd ymm15, ymm6, ymm13
  vfmadd231pd ymm0, ymm7, ymm14
  vaddpd ymm0, ymm0, ymm15
  vbroadcastsd ymm1, qword [r10 + 56]; A[i][k+7]
  lea rax, [rsi + r8 * 4]
  lea rax, [rax + r8 * 2]
  add rax, r8
  vmovupd ymm2, [rax] ; B[k+7][j...j+3]
  vfmadd231pd ymm0, ymm1, ymm2
  dec r9
  jz INNER_LOOP_DONE
  add r10, 64
  lea rsi, [rsi + r8 * 8] ; go to B[k + 8][j ... j+3]
  jmp INNER_LOOP
INNER_LOOP_DONE:
  sub r10, 192
  add r10, rcx ; (start r10 at A[i+1][A_start_col])
  shl r8, 3   ; (get 8 B row stride amount)
  sub rsi, r8 ; (reset position in B to B[A_start_col][j])
  sub rsi, r8
  sub rsi, r8
  shr r8, 3 ; set r8 back to B row stride amount
  vmovupd [rdx], ymm0 ; store result into C
  add rdx, r8 ; (start C at C[i+1][j])
  inc r11
  cmp r11, 32;
  jl MIDDLE_LOOP
MIDDLE_LOOP_DONE:
  shl r8, 5 ; (get 32 C row stride amount)
  sub rdx, r8 ; (go back to start)
  shr r8, 5 ; (set r8 back to 1 row stride)
  add rdx, 32; (advance C to next 4 columns)
  add rsi, 32; (advance B to next 4 columns)
  dec r12
  jz DONE
  jmp OUTER_LOOP

DONE:
  pop r12
  ret



