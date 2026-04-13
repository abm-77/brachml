.section ".text.boot"
.global _start

_start:
    // only let core 0 run, park the rest
    mrs     x0, mpidr_el1
    and     x0, x0, #0x3
    cbnz    x0, halt

    // disable SIMD/FP traps at EL2 via CPTR_EL2
    // clear bit 10 (TFP) to allow NEON/FP at EL2 and EL0
    mrs     x0, cptr_el2
    bic     x0, x0, #(1 << 10)
    msr     cptr_el2, x0
    isb

    // set up stack — grows downward from load address
    ldr     x0, =_start
    mov     sp, x0

    // zero BSS
    ldr     x0, =__bss_start
    ldr     x1, =__bss_end
    mov     x2, #0
bss_loop:
    cmp     x0, x1
    bge     bss_done
    str     x2, [x0], #8
    b       bss_loop
bss_done:

    // jump to C entry point
    bl      main

halt:
    wfe
    b       halt
