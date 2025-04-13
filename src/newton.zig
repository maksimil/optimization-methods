const config = @import("config.zig");
const gd = @import("gradient_descent.zig");

const Scalar = config.Scalar;
const Vec2 = config.Vec2;
const Mat2 = config.Mat2;

const kGradTol: Scalar = 1e-8;
const kIterMax: usize = 100_000;
const kMinEigen: Scalar = 1e-6;

// add identity so that all eigenvalues are above kMinEigen
fn ModifyHessian(h: Mat2) Mat2 {
    const h00 = h[0][0];
    const h11 = h[1][1];
    const h01 = h[0][1];

    const trace = h00 + h11;
    const lambdamin = (trace - @sqrt(trace * trace + 4 * h01 * h01)) / 2;
    const tau = @max(0, kMinEigen - lambdamin);

    return .{ .{ h00 + tau, h01 }, .{ h01, h11 + tau } };
}

fn Solve2(h: Mat2, x: Vec2) Vec2 {
    const h00 = h[0][0];
    const h11 = h[1][1];
    const h01 = h[0][1];

    const d0 = h00;
    const l = h01 / d0;
    const d1 = h11 - l * l * d0;

    var r = x;

    r[1] -= l * r[0];
    r[0] /= d0;
    r[1] /= d1;
    r[0] -= l * r[1];

    return r;
}

pub fn Optimize(f: anytype, grad: anytype, hess: anytype, x0: Vec2) Vec2 {
    var x = x0;
    var gx = Vec2{ 1, 1 };
    var alpha: Scalar = 1.0;

    var niter: usize = 0;

    while (niter < kIterMax and
        config.Norm1(gx) > kGradTol and
        alpha != 0)
    {
        gx = grad.call(x);
        const h = ModifyHessian(hess.call(x));
        const p = Solve2(h, gx);

        const ls = gd.LineSearch(f, 1.0, x, .{ -p[0], -p[1] }, gx);

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
