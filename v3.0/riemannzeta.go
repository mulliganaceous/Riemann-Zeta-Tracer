package main

import (
	"fmt"
	"math/cmplx"
	"time"
)

const TERMS uint = 65536

func zeta(s complex128) complex128 {
	var denom complex128 = complex(1, 0) - cmplx.Pow(complex(2, 0), (complex(1, 0)-s))
	return eta(s) / denom
}

func zeta_multithreaded(s complex128, threads uint8) complex128 {
	var denom complex128 = complex(1, 0) - cmplx.Pow(complex(2, 0), (complex(1, 0)-s))
	return eta_multithreaded(s, threads) / denom
}

func eta(s complex128) complex128 {
	if real(s) == 1 && imag(s) == 1 {
		return cmplx.Inf()
	}
	var sum complex128 = complex(0, 0)
	for k := uint(1); k <= TERMS; k++ {
		sum += etaterm(s, k)
	}
	return sum + etaterm(s, TERMS+1)/2
}

func eta_multithreaded(s complex128, threads uint8) complex128 {
	c := make(chan complex128)
	var subterms uint = (uint(threads) + TERMS - 1) / uint(threads)
	var totalterms uint = subterms * uint(threads)
	var ini uint = 0
	for t := uint8(0); t < threads; t++ {
		go eta_slice(ini+1, ini+subterms, s, c)
		ini += subterms
	}
	var sum complex128 = complex(0, 0)
	for t := uint8(0); t < threads; t++ {
		sum += <-c
	}

	return sum + etaterm(s, totalterms+1)/2
}

func eta_slice(ini uint, fin uint, s complex128, c chan complex128) {
	var sum complex128 = complex(0, 0)
	for k := ini; k <= fin; k++ {
		sum += etaterm(s, k)
	}
	c <- sum
}

func etaterm(s complex128, k uint) complex128 {
	var sgn int = 1
	if k&1 == 0 {
		sgn = -1
	}
	var ans complex128 = complex(float64(sgn), 0) * cmplx.Pow(complex(float64(k), 0), -s)
	return ans
}

func main() {
	/* Hello World */
	fmt.Println("Hello, World!")
	/* Input imaginary number */
	var a, b float64
	fmt.Println("Enter real component: ")
	fmt.Scanf("%g", &a)
	fmt.Println("Enter imag component: ")
	fmt.Scanf("%g", &b)
	var z complex128 = complex(a, b)
	fmt.Printf("z = %g\n", z)
	/* Obtain zeta function */
	const THREADS = 12
	fmt.Printf("zeta(z) = %g\n", zeta(z))
	fmt.Printf("zeta(z) = %g\n", zeta_multithreaded(z, THREADS))
	/* Time loop for the first 360 zeta */
	const INTERVAL uint = 60
	const TOPHEIGHT = 360
	var t1 = time.Now()
	for j := uint(0); j <= INTERVAL*TOPHEIGHT; j++ {
		var height float64 = float64(j/INTERVAL) + float64(j%INTERVAL)/float64(INTERVAL)
		zs := zeta(complex(0.5, height))
		if j%INTERVAL == 0 {
			fmt.Printf("zeta(0.5 + %gi) = %g\n", height, zs)
		}
	}
	fmt.Printf("Single-threaded time: %s\n", time.Since(t1))
	t1 = time.Now()
	for j := uint(0); j <= INTERVAL*TOPHEIGHT; j++ {
		var height float64 = float64(j/INTERVAL) + float64(j%INTERVAL)/float64(INTERVAL)
		zs := zeta_multithreaded(complex(0.5, height), THREADS)
		if j%INTERVAL == 0 {
			fmt.Printf("zeta(0.5 + %gi) = %g\n", height, zs)
		}
	}
	fmt.Printf("Single-threaded time: %s\n", time.Since(t1))
}
