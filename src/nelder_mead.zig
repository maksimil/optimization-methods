const config = @import("config.zig");
const std = @import("std");

const Vec2 = config.Vec2;
const Scalar = config.Scalar;

const kDelt0: Scalar = 1e-2;
const kDelt1: Scalar = 1e-1;
const kIterMax: usize = 1000;
const kXTol: Scalar = 1e-10;

fn ModifyCoord(x: Scalar) Scalar {
    if (x >= 0) {
        return (1 + kDelt1) * x + kDelt0;
    } else {
        return (1 + kDelt1) * x - kDelt0;
    }
}

fn SwapSimplex(simplex: *[3]Vec2, values: *[3]Scalar, i: usize, j: usize) void {
    std.mem.swap(Vec2, &simplex[i], &simplex[j]);
    std.mem.swap(Scalar, &values[i], &values[j]);
}

fn SwapIf(simplex: *[3]Vec2, values: *[3]Scalar, i: usize, j: usize) void {
    if (values[i] > values[j]) {
        SwapSimplex(simplex, values, i, j);
    }
}

fn SortCoords(simplex: *[3]Vec2, values: *[3]Scalar) void {
    SwapIf(simplex, values, 0, 2);
    SwapIf(simplex, values, 1, 2);
    SwapIf(simplex, values, 0, 1);
}

fn SimplexSize(simplex: [3]Vec2) Scalar {
    return @max(
        @abs(simplex[0][0] - simplex[1][0]),
        @abs(simplex[0][1] - simplex[1][1]),
        @abs(simplex[0][0] - simplex[2][0]),
        @abs(simplex[0][1] - simplex[2][1]),
    );
}

fn Reflection(simplex: [3]Vec2, t: Scalar) Vec2 {
    const centroid = .{
        (simplex[0][0] + simplex[1][0]) / 2,
        (simplex[0][1] + simplex[1][1]) / 2,
    };

    return .{
        centroid[0] + t * (simplex[2][0] - centroid[0]),
        centroid[1] + t * (simplex[2][1] - centroid[1]),
    };
}

// f.call : Vec2 -> Scalar
pub fn Optimize(f: anytype, x0: Vec2) Vec2 {

    // --- initial simplex ---

    var simplex: [3]Vec2 = .{ x0, x0, x0 };
    simplex[1][0] = ModifyCoord(simplex[1][0]);
    simplex[2][1] = ModifyCoord(simplex[2][1]);

    var values: [3]Scalar = .{
        f.call(simplex[0]),
        f.call(simplex[1]),
        f.call(simplex[2]),
    };

    SortCoords(&simplex, &values);

    // config.stdout.print(
    //     "iter=0\nsimplex={any}\nvalues={any}\n",
    //     .{ simplex, values },
    // ) catch unreachable;

    // --- the loop ---

    var niter: usize = 0;

    while (niter < kIterMax and SimplexSize(simplex) > kXTol) {
        const refl = Reflection(simplex, -1);
        const frefl = f.call(refl);

        if (values[0] <= frefl and frefl < values[1]) {
            simplex[2] = refl;
            values[2] = frefl;

            SwapSimplex(&simplex, &values, 1, 2);
        } else if (frefl < values[0]) {
            const drefl = Reflection(simplex, -2);
            const fdrefl = f.call(drefl);

            var r = refl;
            var fr = frefl;

            if (fdrefl < frefl) {
                r = drefl;
                fr = fdrefl;
            }

            simplex[2] = r;
            values[2] = fr;

            SwapSimplex(&simplex, &values, 0, 2);
            SwapSimplex(&simplex, &values, 1, 2);
        } else {
            std.debug.assert(frefl >= values[1]);

            var contracted = false;

            if (frefl < values[2]) {
                const hrefl = Reflection(simplex, -0.5);
                const fhrefl = f.call(hrefl);

                if (fhrefl < frefl) {
                    simplex[2] = hrefl;
                    values[2] = fhrefl;

                    contracted = true;
                    SortCoords(&simplex, &values);
                }
            } else {
                const hrefl = Reflection(simplex, 0.5);
                const fhrefl = f.call(hrefl);

                if (fhrefl < values[2]) {
                    simplex[2] = hrefl;
                    values[2] = fhrefl;

                    contracted = true;
                    SortCoords(&simplex, &values);
                }
            }

            if (!contracted) {
                simplex[1][0] = (simplex[0][0] + simplex[1][0]) / 2;
                simplex[1][1] = (simplex[0][1] + simplex[1][1]) / 2;
                simplex[2][0] = (simplex[0][0] + simplex[2][0]) / 2;
                simplex[2][1] = (simplex[0][1] + simplex[2][1]) / 2;

                values[1] = f.call(simplex[1]);
                values[2] = f.call(simplex[2]);

                SortCoords(&simplex, &values);
            }
        }

        niter += 1;

        // config.stdout.print(
        //     "iter={d}\nsimplex={any}\nvalues={any}\n",
        //     .{ niter, simplex, values },
        // ) catch unreachable;
    }

    return simplex[0];
}
