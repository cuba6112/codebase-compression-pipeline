#!/usr/bin/env node
/**
 * TypeScript AST Parser for Codebase Compression Pipeline
 * 
 * Uses @typescript-eslint/parser to extract detailed AST information
 * from TypeScript and JavaScript files.
 */

const fs = require('fs');
const path = require('path');

// Check if required packages are installed
let parser;
try {
    parser = require('@typescript-eslint/parser');
} catch (error) {
    console.error(JSON.stringify({
        error: "Missing dependencies. Please run: npm install -g @typescript-eslint/parser @typescript-eslint/typescript-estree typescript"
    }));
    process.exit(1);
}

/**
 * Parse TypeScript/JavaScript file and extract metadata
 */
function parseFile(filePath, content) {
    try {
        // Parse options
        const parserOptions = {
            sourceType: 'module',
            ecmaVersion: 'latest',
            ecmaFeatures: {
                jsx: true,
                globalReturn: false
            },
            // Enable all TypeScript features
            project: false,  // Don't require tsconfig
            tsconfigRootDir: undefined,
            extraFileExtensions: ['.tsx', '.jsx'],
            warnOnUnsupportedTypeScriptVersion: false
        };

        // Parse the file
        const ast = parser.parse(content, parserOptions);
        
        // Extract metadata
        const metadata = {
            imports: [],
            exports: [],
            functions: [],
            classes: [],
            interfaces: [],
            types: [],
            enums: [],
            variables: [],
            complexity: 1
        };

        // Walk the AST
        walkAST(ast, metadata);

        return {
            success: true,
            data: metadata
        };

    } catch (error) {
        return {
            success: false,
            error: error.message,
            line: error.lineNumber || null
        };
    }
}

/**
 * Walk AST and extract information
 */
function walkAST(node, metadata, depth = 0) {
    if (!node || depth > 100) return;

    switch (node.type) {
        // Imports
        case 'ImportDeclaration':
            extractImport(node, metadata);
            break;

        // Exports
        case 'ExportNamedDeclaration':
        case 'ExportDefaultDeclaration':
        case 'ExportAllDeclaration':
            extractExport(node, metadata);
            break;

        // Functions
        case 'FunctionDeclaration':
        case 'FunctionExpression':
        case 'ArrowFunctionExpression':
            extractFunction(node, metadata);
            break;

        // Classes
        case 'ClassDeclaration':
        case 'ClassExpression':
            extractClass(node, metadata);
            break;

        // TypeScript specific
        case 'TSInterfaceDeclaration':
            extractInterface(node, metadata);
            break;

        case 'TSTypeAliasDeclaration':
            extractTypeAlias(node, metadata);
            break;

        case 'TSEnumDeclaration':
            extractEnum(node, metadata);
            break;

        // Variables
        case 'VariableDeclaration':
            extractVariables(node, metadata);
            break;

        // Control flow (for complexity)
        case 'IfStatement':
        case 'SwitchStatement':
        case 'WhileStatement':
        case 'DoWhileStatement':
        case 'ForStatement':
        case 'ForInStatement':
        case 'ForOfStatement':
        case 'ConditionalExpression':
        case 'LogicalExpression':
            metadata.complexity++;
            break;
    }

    // Recursively walk child nodes
    for (const key in node) {
        if (node[key] && typeof node[key] === 'object') {
            if (Array.isArray(node[key])) {
                node[key].forEach(child => walkAST(child, metadata, depth + 1));
            } else {
                walkAST(node[key], metadata, depth + 1);
            }
        }
    }
}

function extractImport(node, metadata) {
    const importInfo = {
        source: node.source.value,
        specifiers: [],
        line: node.loc?.start.line
    };

    if (node.specifiers) {
        node.specifiers.forEach(spec => {
            if (spec.type === 'ImportDefaultSpecifier') {
                importInfo.specifiers.push({
                    type: 'default',
                    name: spec.local.name
                });
            } else if (spec.type === 'ImportSpecifier') {
                importInfo.specifiers.push({
                    type: 'named',
                    imported: spec.imported.name,
                    local: spec.local.name
                });
            } else if (spec.type === 'ImportNamespaceSpecifier') {
                importInfo.specifiers.push({
                    type: 'namespace',
                    name: spec.local.name
                });
            }
        });
    }

    metadata.imports.push(importInfo);
}

function extractExport(node, metadata) {
    const exportInfo = {
        type: node.type,
        line: node.loc?.start.line
    };

    if (node.type === 'ExportDefaultDeclaration') {
        exportInfo.default = true;
        if (node.declaration?.id?.name) {
            exportInfo.name = node.declaration.id.name;
        }
    } else if (node.type === 'ExportNamedDeclaration') {
        exportInfo.names = [];
        if (node.specifiers) {
            node.specifiers.forEach(spec => {
                exportInfo.names.push({
                    exported: spec.exported.name,
                    local: spec.local?.name
                });
            });
        }
        if (node.declaration) {
            walkAST(node.declaration, metadata);
        }
    }

    metadata.exports.push(exportInfo);
}

function extractFunction(node, metadata) {
    const funcInfo = {
        name: node.id?.name || '<anonymous>',
        type: node.type,
        async: node.async || false,
        generator: node.generator || false,
        params: [],
        line: node.loc?.start.line,
        returnType: null
    };

    // Extract parameters
    if (node.params) {
        node.params.forEach(param => {
            const paramInfo = {
                name: param.name || param.left?.name || '<destructured>',
                type: param.typeAnnotation?.typeAnnotation?.type,
                optional: param.optional || false,
                default: param.right ? true : false
            };
            funcInfo.params.push(paramInfo);
        });
    }

    // Extract return type
    if (node.returnType?.typeAnnotation) {
        funcInfo.returnType = node.returnType.typeAnnotation.type;
    }

    metadata.functions.push(funcInfo);
}

function extractClass(node, metadata) {
    const classInfo = {
        name: node.id?.name || '<anonymous>',
        extends: node.superClass?.name,
        implements: [],
        methods: [],
        properties: [],
        abstract: node.abstract || false,
        line: node.loc?.start.line
    };

    // Extract implements
    if (node.implements) {
        node.implements.forEach(impl => {
            classInfo.implements.push(impl.expression.name);
        });
    }

    // Extract class body
    if (node.body?.body) {
        node.body.body.forEach(member => {
            if (member.type === 'MethodDefinition') {
                classInfo.methods.push({
                    name: member.key.name,
                    kind: member.kind,
                    static: member.static || false,
                    private: member.accessibility === 'private',
                    protected: member.accessibility === 'protected',
                    async: member.value?.async || false
                });
            } else if (member.type === 'PropertyDefinition') {
                classInfo.properties.push({
                    name: member.key?.name || '<computed>',
                    static: member.static || false,
                    private: member.accessibility === 'private',
                    protected: member.accessibility === 'protected',
                    readonly: member.readonly || false,
                    optional: member.optional || false,
                    type: member.typeAnnotation?.typeAnnotation?.type
                });
            }
        });
    }

    metadata.classes.push(classInfo);
}

function extractInterface(node, metadata) {
    const interfaceInfo = {
        name: node.id.name,
        extends: [],
        members: [],
        line: node.loc?.start.line
    };

    // Extract extends
    if (node.extends) {
        node.extends.forEach(ext => {
            interfaceInfo.extends.push(ext.expression.name);
        });
    }

    // Extract members
    if (node.body?.body) {
        node.body.body.forEach(member => {
            const memberInfo = {
                name: member.key?.name || '<computed>',
                type: member.type,
                optional: member.optional || false,
                readonly: member.readonly || false
            };

            if (member.typeAnnotation) {
                memberInfo.valueType = member.typeAnnotation.typeAnnotation.type;
            }

            interfaceInfo.members.push(memberInfo);
        });
    }

    metadata.interfaces.push(interfaceInfo);
}

function extractTypeAlias(node, metadata) {
    const typeInfo = {
        name: node.id.name,
        type: node.typeAnnotation?.type,
        line: node.loc?.start.line
    };

    metadata.types.push(typeInfo);
}

function extractEnum(node, metadata) {
    const enumInfo = {
        name: node.id.name,
        const: node.const || false,
        members: [],
        line: node.loc?.start.line
    };

    if (node.members) {
        node.members.forEach(member => {
            enumInfo.members.push({
                name: member.id.name,
                value: member.initializer?.value
            });
        });
    }

    metadata.enums.push(enumInfo);
}

function extractVariables(node, metadata) {
    node.declarations.forEach(decl => {
        if (decl.id?.name) {
            const varInfo = {
                name: decl.id.name,
                kind: node.kind, // const, let, var
                type: decl.id.typeAnnotation?.typeAnnotation?.type,
                initialized: decl.init !== null,
                line: node.loc?.start.line
            };
            metadata.variables.push(varInfo);
        }
    });
}

// Main execution
if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        console.error(JSON.stringify({
            error: "Usage: node typescript_ast_parser.js <file_path>"
        }));
        process.exit(1);
    }

    const filePath = args[0];
    
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const result = parseFile(filePath, content);
        
        // Output JSON result
        console.log(JSON.stringify(result, null, 2));
        
    } catch (error) {
        console.error(JSON.stringify({
            success: false,
            error: error.message
        }));
        process.exit(1);
    }
}

module.exports = { parseFile };