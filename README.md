![t0+30](./logo1.png)
# Riemann Zeta CUDA tracer

This is **CUDA** and **C++** program for the accelerated and high-precision calculation and visualization of the Riemann Zeta function. The goal of this 

![t7000+5](./logo2.png)

* **Highlight:** https://www.youtube.com/watch?v=6XrqDJFJR5k
* **SoME4 Submission:** https://youtu.be/702X-GXbGs8?si=bLDanf7dLGf2OEdZ
* **Livestream (1000):** https://www.youtube.com/watch?v=zm4WsYN9bz4

## How to Run

As of now, the application, though functional, needs to consider the various CUDA architectures to ensure maximum compatibility.

OpenCV is a required package to be downloaded and linked.

Performance and testing is currently the main priority for future updates.

## Images

This version is the one used for the [#SoME4](https://some.3b1b.co/entries/ab36e65b-c4ad-4ae0-a31f-36275cbdf212) contest.
![h07000.png](./images/v4.0/h07000.png)

## Legacy versions
Development of the Riemann Zeta Tracer started strictly as a hobby project in Q2 2018 (man, time runs fast). The project was revived in 2024 from the first version as part of **Operation: Hashling**, specifically for triggering the assignment of unique usernames to the remaining users. A Go version was planned to be built, but was superceded in favor of the C++ and CUDA version.

* The unmarked version animates the Riemann zeta function live from height zero.
* The 7005 edition animates the Riemann zeta function from height 7005.
    * This is the famous Lehmer pair.
        * $\gamma_{6709} \approx 7005.0628661749205813803437835888415$
        * $\gamma_{6710} \approx 7005.1005646726467215687204319795170$
    * The two zeroes are separated by a distance of only $\Delta \gamma_{6709} \approx 0.038$.
        * $\Delta \gamma_{6709} \approx 0.0386984977261401883766483906755 \approx 25.840^{-1}$.
        * This is between a twenty-fifth to a twenty-sixth of a unit interval.
    * The two zeroes also have an arc length of $8.131 \times 10^{-3}$ units, the shortest.
    * A comparable scenario occurs earlier at height $5229$.
        * Both the distance and arc length are slightly longer.
* The FFFF edition does not animate the Riemann zeta function, but it calculates the entire path up to height 65536 with resolution 1024 and accuracy level 65536.
    * It will output a 4K version of the image at the end.

While the application is running, the console will output every time it increments by one imaginary unit, when it is about to detect a zero, and when it detects a zero. Additionally, the console will output when new records are being set. These include:

* Farthest distance from origin
* Fastest speed of the curve
* Closest pair of zeta zeroes
* Shortest and longest pair of zeta zeroes in terms of arc length.

This information is also shown live on the GUI, which is currently scaled at 720p.

### Legacy version I (Swing)

This version was not officially released. It originated as a 2018 project, and was used as my main profile picture. I have put the project on hiatus in 2019, for four bloody yearrs.
|Image                                 |Description
|--------------------------------------|-----------
|![h65536](./images/v1.0/Riemann65536.gif)  |Riemann zeta limaçon up to height 65536.
|![h14400](./images/v1.0/Riemann14400.gif)  |Closeup of the limaçon up to height 14400.
|![h65536a](./images/v1.0/Riemann65536a.gif)|Closeup of the final segment of the limaçon up to height 65536. The scale is the same as the height 14400 limacon; the circle is the unit circle, and the cross has radius 1/8.
|![anim1](./images/v1.0/RiemannZeta1.gif)   |Animation of the first segment of the curve with step size 1/2 and resolution 1/64.
|![anim2](./images/v1.0/Valentine19.webp)   |Fast animation of the curve. 

### Legacy Version II (Swing)

This is currently basic **Java Swing** program which animates the path of the Riemann zeta function on the critical line. Unitl then, there aren't many visualizations of the zeta function at high heights, let alone entire livestreams of it.

The trailing part of the path is colored red, while the leading part of the path is colored violet. Below is the the last 1000 parts of the zeta spiral before we reach one million.

This is the program that runs for the [**Operation: Hashling**](https://discordrollout.nekos.sh/) [livestream](https://youtu.be/o-XUeCMx5s0?feature=shared), where Discord starts assigning unique usernames to the remaining users who have not updated their usernames themselves.

Java Swing is now considered legacy..

|Image                                 |Description
|--------------------------------------|-----------
|![h7021](./images/v2.0/Riemann7021.png)    |Riemann zeta limaçon from the Lehmer pair and 16 unit after. The Lehmer arc is not visible at this scale, even at 4K.
|![h10000](./images/v2.0/Riemann0x10000.png)  |Riemann zeta limaçon up to height 65536 (last 1024 units)

## Resources

* [Andrew Odlyzko: Tables of zeros of the Riemann zeta function](https://www-users.cse.umn.edu/~odlyzko/zeta_tables/index.html)
    * Contains downloads of the first hundred zeroes accurate to over 1000 decimal places.
    * Contains downloads of zeroes at zettascale heights.
* [LMFDB](https://www.lmfdb.org/zeros/zeta/)
    * A database containing the first hundred billion zeroes of the Riemann zeta function above the real line, with thirty-digit precision.
