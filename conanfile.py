from conans import ConanFile, CMake


class QuestiaConan(ConanFile):
    name = "Questia"
    version = "0.0.1"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    license = "https://github.com/SquireGames/Questia/blob/master/LICENSE.txt"
    url = "https://github.com/SquireGames/Questia"
    default_options = "shared=False"
    generators = "cmake"
    exports_sources = "cmake*", "doc*", "include*", "samples*", "src*", "test*", "CMakeLists.txt"
    requires = "qeng/0.0.1@squiregames/testing"

    def configure(self):
        for req in self.requires:
            self.options[req.split("/", 1)[0]].shared = self.options.shared
        
        # ensure proper compiler settings when using Visual Studio
        if self.settings.compiler == "Visual Studio":
            if self.options.shared and self.settings.build_type == "Debug" and self.settings.compiler.runtime != "MDd":
                self.output.warn("Use '-s compiler.runtime=MDd' when compiling with shared=true and build_type=Debug")
            elif self.options.shared and self.settings.build_type == "Release" and self.settings.compiler.runtime != "MD":
                self.output.warn("Use '-s compiler.runtime=MD' when compiling with shared=true and build_type=Release")
            elif not self.options.shared and self.settings.build_type == "Debug" and self.settings.compiler.runtime != "MTd":
                self.output.warn("Use '-s compiler.runtime=MTd' when compiling with shared=false and build_type=Debug")
            elif not self.options.shared and self.settings.build_type == "Release" and self.settings.compiler.runtime != "MT":
                self.output.warn("Use '-s compiler.runtime=MT' when compiling with shared=false and build_type=Release")
    
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        self.copy("*.h", dst="include", src="src")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.dylib*", dst="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
