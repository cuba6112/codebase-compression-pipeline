package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"strings"
)

// Metadata represents the extracted metadata from Go source code
type Metadata struct {
	Success     bool              `json:"success"`
	Error       string            `json:"error,omitempty"`
	Data        *ExtractedData    `json:"data,omitempty"`
}

// ExtractedData contains all extracted information
type ExtractedData struct {
	Package     string            `json:"package"`
	Imports     []ImportInfo      `json:"imports"`
	Functions   []FunctionInfo    `json:"functions"`
	Types       []TypeInfo        `json:"types"`
	Interfaces  []InterfaceInfo   `json:"interfaces"`
	Variables   []VariableInfo    `json:"variables"`
	Constants   []ConstantInfo    `json:"constants"`
	Complexity  int               `json:"complexity"`
}

// ImportInfo represents an import declaration
type ImportInfo struct {
	Path  string `json:"path"`
	Alias string `json:"alias,omitempty"`
	Line  int    `json:"line"`
}

// FunctionInfo represents a function or method
type FunctionInfo struct {
	Name       string      `json:"name"`
	Receiver   string      `json:"receiver,omitempty"`
	Params     []ParamInfo `json:"params"`
	Results    []string    `json:"results"`
	IsExported bool        `json:"is_exported"`
	Line       int         `json:"line"`
}

// ParamInfo represents a function parameter
type ParamInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// TypeInfo represents a type declaration
type TypeInfo struct {
	Name       string          `json:"name"`
	Type       string          `json:"type"`
	Fields     []FieldInfo     `json:"fields,omitempty"`
	Methods    []FunctionInfo  `json:"methods,omitempty"`
	IsExported bool            `json:"is_exported"`
	Line       int             `json:"line"`
}

// FieldInfo represents a struct field
type FieldInfo struct {
	Name       string   `json:"name"`
	Type       string   `json:"type"`
	Tags       []string `json:"tags,omitempty"`
	IsExported bool     `json:"is_exported"`
}

// InterfaceInfo represents an interface declaration
type InterfaceInfo struct {
	Name       string         `json:"name"`
	Methods    []MethodInfo   `json:"methods"`
	IsExported bool           `json:"is_exported"`
	Line       int            `json:"line"`
}

// MethodInfo represents an interface method
type MethodInfo struct {
	Name    string   `json:"name"`
	Params  []string `json:"params"`
	Results []string `json:"results"`
}

// VariableInfo represents a variable declaration
type VariableInfo struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	IsExported bool   `json:"is_exported"`
	Line       int    `json:"line"`
}

// ConstantInfo represents a constant declaration
type ConstantInfo struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	Value      string `json:"value,omitempty"`
	IsExported bool   `json:"is_exported"`
	Line       int    `json:"line"`
}

func main() {
	if len(os.Args) < 2 {
		result := Metadata{
			Success: false,
			Error:   "Usage: go_ast_parser <file_path>",
		}
		json.NewEncoder(os.Stdout).Encode(result)
		os.Exit(1)
	}

	filePath := os.Args[1]
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		result := Metadata{
			Success: false,
			Error:   fmt.Sprintf("Failed to read file: %v", err),
		}
		json.NewEncoder(os.Stdout).Encode(result)
		os.Exit(1)
	}

	data, err := parseGoFile(filePath, string(content))
	if err != nil {
		result := Metadata{
			Success: false,
			Error:   fmt.Sprintf("Failed to parse file: %v", err),
		}
		json.NewEncoder(os.Stdout).Encode(result)
		os.Exit(1)
	}

	result := Metadata{
		Success: true,
		Data:    data,
	}
	json.NewEncoder(os.Stdout).Encode(result)
}

func parseGoFile(filePath string, content string) (*ExtractedData, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, content, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	data := &ExtractedData{
		Package:    node.Name.Name,
		Imports:    []ImportInfo{},
		Functions:  []FunctionInfo{},
		Types:      []TypeInfo{},
		Interfaces: []InterfaceInfo{},
		Variables:  []VariableInfo{},
		Constants:  []ConstantInfo{},
		Complexity: 1,
	}

	// Extract imports
	for _, imp := range node.Imports {
		importInfo := ImportInfo{
			Path: strings.Trim(imp.Path.Value, "\""),
			Line: fset.Position(imp.Pos()).Line,
		}
		if imp.Name != nil {
			importInfo.Alias = imp.Name.Name
		}
		data.Imports = append(data.Imports, importInfo)
	}

	// Walk the AST
	ast.Inspect(node, func(n ast.Node) bool {
		switch decl := n.(type) {
		case *ast.FuncDecl:
			data.Functions = append(data.Functions, extractFunction(decl, fset))
			// Count complexity
			countComplexity(decl.Body, &data.Complexity)
		
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					switch t := s.Type.(type) {
					case *ast.InterfaceType:
						data.Interfaces = append(data.Interfaces, extractInterface(s, t, fset))
					case *ast.StructType:
						typeInfo := extractStruct(s, t, fset)
						data.Types = append(data.Types, typeInfo)
					default:
						typeInfo := TypeInfo{
							Name:       s.Name.Name,
							Type:       fmt.Sprintf("%T", s.Type),
							IsExported: ast.IsExported(s.Name.Name),
							Line:       fset.Position(s.Pos()).Line,
						}
						data.Types = append(data.Types, typeInfo)
					}
				case *ast.ValueSpec:
					if decl.Tok == token.CONST {
						for i, name := range s.Names {
							constInfo := ConstantInfo{
								Name:       name.Name,
								IsExported: ast.IsExported(name.Name),
								Line:       fset.Position(name.Pos()).Line,
							}
							if s.Type != nil {
								constInfo.Type = fmt.Sprintf("%s", s.Type)
							}
							if i < len(s.Values) {
								constInfo.Value = fmt.Sprintf("%s", s.Values[i])
							}
							data.Constants = append(data.Constants, constInfo)
						}
					} else if decl.Tok == token.VAR {
						for _, name := range s.Names {
							varInfo := VariableInfo{
								Name:       name.Name,
								IsExported: ast.IsExported(name.Name),
								Line:       fset.Position(name.Pos()).Line,
							}
							if s.Type != nil {
								varInfo.Type = fmt.Sprintf("%s", s.Type)
							}
							data.Variables = append(data.Variables, varInfo)
						}
					}
				}
			}
		}
		return true
	})

	// Find methods for types
	for _, fn := range data.Functions {
		if fn.Receiver != "" {
			for i := range data.Types {
				if data.Types[i].Name == fn.Receiver {
					data.Types[i].Methods = append(data.Types[i].Methods, fn)
				}
			}
		}
	}

	return data, nil
}

func extractFunction(fn *ast.FuncDecl, fset *token.FileSet) FunctionInfo {
	info := FunctionInfo{
		Name:       fn.Name.Name,
		IsExported: ast.IsExported(fn.Name.Name),
		Line:       fset.Position(fn.Pos()).Line,
		Params:     []ParamInfo{},
		Results:    []string{},
	}

	// Extract receiver
	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		recv := fn.Recv.List[0]
		switch t := recv.Type.(type) {
		case *ast.StarExpr:
			if ident, ok := t.X.(*ast.Ident); ok {
				info.Receiver = ident.Name
			}
		case *ast.Ident:
			info.Receiver = t.Name
		}
	}

	// Extract parameters
	if fn.Type.Params != nil {
		for _, field := range fn.Type.Params.List {
			paramType := fmt.Sprintf("%s", field.Type)
			for _, name := range field.Names {
				info.Params = append(info.Params, ParamInfo{
					Name: name.Name,
					Type: paramType,
				})
			}
			// Handle unnamed parameters
			if len(field.Names) == 0 {
				info.Params = append(info.Params, ParamInfo{
					Name: "",
					Type: paramType,
				})
			}
		}
	}

	// Extract results
	if fn.Type.Results != nil {
		for _, field := range fn.Type.Results.List {
			resultType := fmt.Sprintf("%s", field.Type)
			info.Results = append(info.Results, resultType)
		}
	}

	return info
}

func extractInterface(spec *ast.TypeSpec, iface *ast.InterfaceType, fset *token.FileSet) InterfaceInfo {
	info := InterfaceInfo{
		Name:       spec.Name.Name,
		IsExported: ast.IsExported(spec.Name.Name),
		Line:       fset.Position(spec.Pos()).Line,
		Methods:    []MethodInfo{},
	}

	if iface.Methods != nil {
		for _, method := range iface.Methods.List {
			if fn, ok := method.Type.(*ast.FuncType); ok {
				for _, name := range method.Names {
					methodInfo := MethodInfo{
						Name:    name.Name,
						Params:  []string{},
						Results: []string{},
					}
					
					// Extract params
					if fn.Params != nil {
						for _, param := range fn.Params.List {
							methodInfo.Params = append(methodInfo.Params, fmt.Sprintf("%s", param.Type))
						}
					}
					
					// Extract results
					if fn.Results != nil {
						for _, result := range fn.Results.List {
							methodInfo.Results = append(methodInfo.Results, fmt.Sprintf("%s", result.Type))
						}
					}
					
					info.Methods = append(info.Methods, methodInfo)
				}
			}
		}
	}

	return info
}

func extractStruct(spec *ast.TypeSpec, st *ast.StructType, fset *token.FileSet) TypeInfo {
	info := TypeInfo{
		Name:       spec.Name.Name,
		Type:       "struct",
		IsExported: ast.IsExported(spec.Name.Name),
		Line:       fset.Position(spec.Pos()).Line,
		Fields:     []FieldInfo{},
		Methods:    []FunctionInfo{},
	}

	if st.Fields != nil {
		for _, field := range st.Fields.List {
			fieldType := fmt.Sprintf("%s", field.Type)
			tags := []string{}
			if field.Tag != nil {
				tags = append(tags, field.Tag.Value)
			}
			
			for _, name := range field.Names {
				info.Fields = append(info.Fields, FieldInfo{
					Name:       name.Name,
					Type:       fieldType,
					Tags:       tags,
					IsExported: ast.IsExported(name.Name),
				})
			}
			// Handle embedded fields
			if len(field.Names) == 0 {
				info.Fields = append(info.Fields, FieldInfo{
					Name:       "",
					Type:       fieldType,
					Tags:       tags,
					IsExported: true,
				})
			}
		}
	}

	return info
}

func countComplexity(block *ast.BlockStmt, complexity *int) {
	if block == nil {
		return
	}
	
	ast.Inspect(block, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt:
			*complexity++
		case *ast.CaseClause:
			*complexity++
		}
		return true
	})
}