pub fn make_tranform(translate: [f32; 3], angle: f32, scale: f32) -> [[f32; 4]; 4] {
    let c = angle.cos() * scale;
    let s = angle.sin() * scale;
    let [dx, dy, dz] = translate;
    [
        [c, 0., s, 0.],
        [0., scale, 0., 0.],
        [-s, 0., c, 0.],
        [dx, dy, dz, 1.],
    ]
}
