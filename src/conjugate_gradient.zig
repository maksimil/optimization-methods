const config = @import("config.zig");
const gd = @import("gradient_descent.zig");

const Vec2 = config.Vec2;
const Scalar = config.Scalar;

const kGradTol: Scalar = 1e-8;
const kIterMax: usize = 100_000;

pub fn Optimize(f: anytype, grad: anytype, x0: Vec2) Vec2 {
    var x = x0;
    var gx = grad.call(x);
    var alpha: Scalar = 1.0;
    var p: Vec2 = .{ 0, 0 };

    var niter: usize = 0;

    while (niter < kIterMax and
        config.Norm1(gx) > kGradTol and
        alpha != 0)
    {
        const gx_prev = gx;
        gx = grad.call(x);
        const gx1 = config.Norm1(gx);

        var betafr: Scalar = 0.0;
        if (@mod(niter, 2) != 0) {
            betafr = config.VSqr(gx) / config.VSqr(gx_prev);
        }

        alpha *= (config.VSqr(gx_prev) / config.Norm1(gx_prev)) /
            (config.VSqr(gx) / gx1);

        p = .{
            -gx[0] + betafr * p[0],
            -gx[1] + betafr * p[1],
        };

        const ls = gd.LineSearch(
            f,
            alpha,
            x,
            .{ p[0] / config.Norm1(p), p[1] / config.Norm1(p) },
            gx,
        );

        alpha = ls.alpha;
        x = ls.x;

        niter += 1;
    }

    if (niter == kIterMax) {
        config.stdout.print("iter max {d} was reached\n", .{kIterMax}) catch unreachable;
    }

    if (alpha == 0) {
        config.stdout.print("line search failed\n", .{}) catch unreachable;
    }

    return x;
}
