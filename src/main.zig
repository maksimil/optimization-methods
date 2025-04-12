const config = @import("config.zig");
const nelder_mead = @import("nelder_mead.zig");

const Vec2 = config.Vec2;
const Scalar = config.Scalar;

const kInitialPoint: Vec2 = .{ 10, 20 };

fn WrapFn(comptime f: anytype) type {
    return struct {
        pub fn call(x: Vec2) Scalar {
            return f(x);
        }
    };
}

pub fn main() !void {
    config.RuntimeInitialize();
    defer config.RuntimeDeinitialize();

    // --- Nelder-Mead method ---
    const res = nelder_mead.Optimize(WrapFn(config.TaskF1), kInitialPoint);
    try config.stdout.print(
        "x={any}, f(x)={e}\n",
        .{ res, config.TaskF1(res) },
    );
}
