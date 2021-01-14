use shaderc::ShaderKind;

pub fn compile_shader(glsl: &str, shader_kind: ShaderKind) -> Vec<u32> {
    use std::io::Cursor;

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    let compiled_file = compiler
        .compile_into_spirv(glsl, shader_kind, "shader.glsl", "main", Some(&options))
        .unwrap();

    let compiled_file = compiled_file.as_binary_u8();

    let spv =
        gfx_auxil::read_spirv(Cursor::new(&compiled_file)).expect("Failed to read spirv file");

    spv
}
