; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

define void @_softmax_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, i32 %4, i32 %5) local_unnamed_addr !dbg !7 {
  %7 = tail call i32 asm "mov.u32 $0, %ctaid.x;", "=r"() #3, !dbg !10
  %8 = tail call i32 asm "mov.u32 $0, %nctaid.x;", "=r"() #3, !dbg !11
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %10 = and i32 %9, 255, !dbg !12
  %11 = or disjoint i32 %10, 256, !dbg !12
  %12 = icmp slt i32 %10, %5, !dbg !13
  %13 = icmp slt i32 %11, %5, !dbg !13
  %14 = icmp slt i32 %7, %4, !dbg !14
  %15 = mul i32 %7, %2, !dbg !15
  %16 = sext i32 %15 to i64, !dbg !16
  %17 = getelementptr float, ptr addrspace(1) %0, i64 %16, !dbg !16
  %18 = zext nneg i32 %10 to i64, !dbg !17
  %19 = getelementptr float, ptr addrspace(1) %17, i64 %18, !dbg !17
  %20 = zext nneg i32 %11 to i64, !dbg !17
  %21 = getelementptr float, ptr addrspace(1) %17, i64 %20, !dbg !17
  %22 = and i1 %12, %14, !dbg !14
  %23 = and i1 %13, %14, !dbg !14
  %24 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %18, !dbg !18
  %25 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %20, !dbg !18
  %26 = select i1 %22, i32 4, i32 0, !dbg !18
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %24, ptr addrspace(1) %19, i32 %26, i1 true) #3, !dbg !18
  %27 = select i1 %23, i32 4, i32 0, !dbg !18
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %25, ptr addrspace(1) %21, i32 %27, i1 true) #3, !dbg !18
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #3, !dbg !18
  br i1 %14, label %.lr.ph, label %._crit_edge, !dbg !14

.lr.ph:                                           ; preds = %6
  %28 = lshr i32 %9, 5, !dbg !12
  %29 = and i32 %9, 31, !dbg !12
  %30 = sub i32 %4, %8
  %31 = icmp eq i32 %29, 0
  %32 = and i32 %28, 7
  %33 = zext nneg i32 %32 to i64
  %34 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %33
  %35 = icmp slt i32 %9, 8
  %36 = sext i32 %9 to i64
  %37 = getelementptr float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %36
  %38 = and i32 %9, 7
  %39 = icmp eq i32 %38, 0
  %40 = and i1 %35, %39
  br label %41, !dbg !14

41:                                               ; preds = %.lr.ph, %41
  %42 = phi i32 [ -1, %.lr.ph ], [ %47, %41 ]
  %43 = phi i32 [ %7, %.lr.ph ], [ %146, %41 ]
  %44 = icmp slt i32 %43, %30, !dbg !14
  %45 = add i32 %42, 1, !dbg !14
  %46 = icmp ugt i32 %42, 2147483646, !dbg !14
  %47 = select i1 %46, i32 %45, i32 0, !dbg !14
  tail call void asm sideeffect "cp.async.wait_group 0x0;", ""() #3, !dbg !18
  tail call void @llvm.nvvm.barrier0(), !dbg !18
  %48 = shl i32 %47, 9, !dbg !18
  %49 = sext i32 %48 to i64, !dbg !18
  %50 = getelementptr float, ptr addrspace(3) @global_smem, i64 %49, !dbg !18
  %51 = getelementptr inbounds float, ptr addrspace(3) %50, i64 %18, !dbg !18
  %52 = load float, ptr addrspace(3) %51, align 4, !dbg !18
  %53 = getelementptr inbounds float, ptr addrspace(3) %50, i64 %20, !dbg !18
  %54 = load float, ptr addrspace(3) %53, align 4, !dbg !18
  %55 = select i1 %12, float %52, float 0xFFF0000000000000, !dbg !18
  %56 = select i1 %13, float %54, float 0xFFF0000000000000, !dbg !18
  %57 = tail call float @llvm.maxnum.f32(float %55, float %56), !dbg !19
  %58 = bitcast float %57 to i32, !dbg !24
  %59 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %58, i32 16, i32 31), !dbg !24
  %60 = bitcast i32 %59 to float, !dbg !24
  %61 = tail call float @llvm.maxnum.f32(float %57, float %60), !dbg !19
  %62 = bitcast float %61 to i32, !dbg !24
  %63 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %62, i32 8, i32 31), !dbg !24
  %64 = bitcast i32 %63 to float, !dbg !24
  %65 = tail call float @llvm.maxnum.f32(float %61, float %64), !dbg !19
  %66 = bitcast float %65 to i32, !dbg !24
  %67 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %66, i32 4, i32 31), !dbg !24
  %68 = bitcast i32 %67 to float, !dbg !24
  %69 = tail call float @llvm.maxnum.f32(float %65, float %68), !dbg !19
  %70 = bitcast float %69 to i32, !dbg !24
  %71 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %70, i32 2, i32 31), !dbg !24
  %72 = bitcast i32 %71 to float, !dbg !24
  %73 = tail call float @llvm.maxnum.f32(float %69, float %72), !dbg !19
  %74 = bitcast float %73 to i32, !dbg !24
  %75 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %74, i32 1, i32 31), !dbg !24
  %76 = bitcast i32 %75 to float, !dbg !24
  %77 = tail call float @llvm.maxnum.f32(float %73, float %76), !dbg !19
  %78 = bitcast float %77 to <1 x i32>, !dbg !24
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %34, <1 x i32> %78, i1 %31) #3, !dbg !24
  tail call void @llvm.nvvm.barrier0(), !dbg !24
  %79 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %37, i1 %35) #3, !dbg !24
  %80 = bitcast i32 %79 to float, !dbg !24
  %81 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %79, i32 4, i32 31), !dbg !24
  %82 = bitcast i32 %81 to float, !dbg !24
  %83 = tail call float @llvm.maxnum.f32(float %80, float %82), !dbg !19
  %84 = bitcast float %83 to i32, !dbg !24
  %85 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %84, i32 2, i32 31), !dbg !24
  %86 = bitcast i32 %85 to float, !dbg !24
  %87 = tail call float @llvm.maxnum.f32(float %83, float %86), !dbg !19
  %88 = bitcast float %87 to i32, !dbg !24
  %89 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %88, i32 1, i32 31), !dbg !24
  %90 = bitcast i32 %89 to float, !dbg !24
  %91 = tail call float @llvm.maxnum.f32(float %87, float %90), !dbg !19
  %92 = bitcast float %91 to <1 x i32>, !dbg !24
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %37, <1 x i32> %92, i1 %40) #3, !dbg !24
  tail call void @llvm.nvvm.barrier0(), !dbg !24
  %93 = load float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), align 16, !dbg !24
  %94 = fsub float %55, %93, !dbg !25
  %95 = fsub float %56, %93, !dbg !25
  %96 = fmul float %94, 0x3FF7154760000000, !dbg !26
  %97 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %96) #3, !dbg !26
  %98 = fmul float %95, 0x3FF7154760000000, !dbg !26
  %99 = tail call float asm "ex2.approx.f32 $0, $1;", "=f,f"(float %98) #3, !dbg !26
  tail call void @llvm.nvvm.barrier0(), !dbg !27
  %100 = fadd float %97, %99, !dbg !29
  %101 = bitcast float %100 to i32, !dbg !27
  %102 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %101, i32 16, i32 31), !dbg !27
  %103 = bitcast i32 %102 to float, !dbg !27
  %104 = fadd float %100, %103, !dbg !29
  %105 = bitcast float %104 to i32, !dbg !27
  %106 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %105, i32 8, i32 31), !dbg !27
  %107 = bitcast i32 %106 to float, !dbg !27
  %108 = fadd float %104, %107, !dbg !29
  %109 = bitcast float %108 to i32, !dbg !27
  %110 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %109, i32 4, i32 31), !dbg !27
  %111 = bitcast i32 %110 to float, !dbg !27
  %112 = fadd float %108, %111, !dbg !29
  %113 = bitcast float %112 to i32, !dbg !27
  %114 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %113, i32 2, i32 31), !dbg !27
  %115 = bitcast i32 %114 to float, !dbg !27
  %116 = fadd float %112, %115, !dbg !29
  %117 = bitcast float %116 to i32, !dbg !27
  %118 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %117, i32 1, i32 31), !dbg !27
  %119 = bitcast i32 %118 to float, !dbg !27
  %120 = fadd float %116, %119, !dbg !29
  %121 = bitcast float %120 to <1 x i32>, !dbg !27
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %34, <1 x i32> %121, i1 %31) #3, !dbg !27
  tail call void @llvm.nvvm.barrier0(), !dbg !27
  %122 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %37, i1 %35) #3, !dbg !27
  %123 = bitcast i32 %122 to float, !dbg !27
  %124 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %122, i32 4, i32 31), !dbg !27
  %125 = bitcast i32 %124 to float, !dbg !27
  %126 = fadd float %123, %125, !dbg !29
  %127 = bitcast float %126 to i32, !dbg !27
  %128 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %127, i32 2, i32 31), !dbg !27
  %129 = bitcast i32 %128 to float, !dbg !27
  %130 = fadd float %126, %129, !dbg !29
  %131 = bitcast float %130 to i32, !dbg !27
  %132 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %131, i32 1, i32 31), !dbg !27
  %133 = bitcast i32 %132 to float, !dbg !27
  %134 = fadd float %130, %133, !dbg !29
  %135 = bitcast float %134 to <1 x i32>, !dbg !27
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %37, <1 x i32> %135, i1 %40) #3, !dbg !27
  tail call void @llvm.nvvm.barrier0(), !dbg !27
  %136 = load float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), align 16, !dbg !27
  %137 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %97, float %136) #3, !dbg !30
  %138 = tail call float asm "div.full.f32 $0, $1, $2;", "=r,r,r"(float %99, float %136) #3, !dbg !30
  %139 = mul i32 %43, %3, !dbg !31
  %140 = sext i32 %139 to i64, !dbg !32
  %141 = getelementptr float, ptr addrspace(1) %1, i64 %140, !dbg !32
  %142 = getelementptr float, ptr addrspace(1) %141, i64 %18, !dbg !33
  %143 = getelementptr float, ptr addrspace(1) %141, i64 %20, !dbg !33
  %144 = bitcast float %137 to i32, !dbg !34
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %144, ptr addrspace(1) %142, i1 %12) #3, !dbg !34
  %145 = bitcast float %138 to i32, !dbg !34
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %145, ptr addrspace(1) %143, i1 %13) #3, !dbg !34
  %146 = add i32 %43, %8, !dbg !14
  %147 = mul i32 %146, %2, !dbg !15
  %148 = sext i32 %147 to i64, !dbg !16
  %149 = getelementptr float, ptr addrspace(1) %0, i64 %148, !dbg !16
  %150 = getelementptr float, ptr addrspace(1) %149, i64 %18, !dbg !17
  %151 = getelementptr float, ptr addrspace(1) %149, i64 %20, !dbg !17
  %152 = and i1 %12, %44, !dbg !14
  %153 = and i1 %13, %44, !dbg !14
  %154 = select i1 %152, i32 4, i32 0, !dbg !18
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %24, ptr addrspace(1) %150, i32 %154, i1 true) #3, !dbg !18
  %155 = select i1 %153, i32 4, i32 0, !dbg !18
  tail call void asm sideeffect "@$3 cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r,b"(ptr addrspace(3) %25, ptr addrspace(1) %151, i32 %155, i1 true) #3, !dbg !18
  tail call void asm sideeffect "cp.async.commit_group ;", ""() #3, !dbg !18
  %156 = icmp slt i32 %146, %4, !dbg !14
  br i1 %156, label %41, label %._crit_edge, !dbg !14

._crit_edge:                                      ; preds = %41, %6
  tail call void asm sideeffect "cp.async.wait_group 0x0;", ""() #3, !dbg !14
  tail call void @llvm.nvvm.barrier0(), !dbg !14
  ret void, !dbg !35
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #2

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "triton_internals_explorer.py", directory: "/home/ubuntu/Learn-Triton/05_fused_softmax/practice")
!4 = !{ptr @_softmax_kernel, !"kernel", i32 1}
!5 = !{ptr @_softmax_kernel, !"reqntidx", i32 256}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = distinct !DISubprogram(name: "_softmax_kernel", linkageName: "_softmax_kernel", scope: !3, file: !3, line: 101, type: !8, scopeLine: 101, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!8 = !DISubroutineType(cc: DW_CC_normal, types: !9)
!9 = !{}
!10 = !DILocation(line: 108, column: 30, scope: !7)
!11 = !DILocation(line: 109, column: 32, scope: !7)
!12 = !DILocation(line: 112, column: 38, scope: !7)
!13 = !DILocation(line: 114, column: 39, scope: !7)
!14 = !DILocation(line: 110, column: 57, scope: !7)
!15 = !DILocation(line: 111, column: 47, scope: !7)
!16 = !DILocation(line: 111, column: 37, scope: !7)
!17 = !DILocation(line: 113, column: 41, scope: !7)
!18 = !DILocation(line: 115, column: 33, scope: !7)
!19 = !DILocation(line: 163, column: 27, scope: !20, inlinedAt: !23)
!20 = distinct !DILexicalBlockFile(scope: !22, file: !21, discriminator: 0)
!21 = !DIFile(filename: "standard.py", directory: "/home/ubuntu/.local/lib/python3.10/site-packages/triton/language")
!22 = distinct !DILexicalBlockFile(scope: !7, file: !21, discriminator: 0)
!23 = !DILocation(line: 116, column: 38, scope: !7)
!24 = !DILocation(line: 184, column: 40, scope: !22, inlinedAt: !23)
!25 = !DILocation(line: 116, column: 31, scope: !7)
!26 = !DILocation(line: 117, column: 32, scope: !7)
!27 = !DILocation(line: 267, column: 36, scope: !22, inlinedAt: !28)
!28 = !DILocation(line: 118, column: 32, scope: !7)
!29 = !DILocation(line: 256, column: 15, scope: !20, inlinedAt: !28)
!30 = !DILocation(line: 119, column: 37, scope: !7)
!31 = !DILocation(line: 120, column: 48, scope: !7)
!32 = !DILocation(line: 120, column: 38, scope: !7)
!33 = !DILocation(line: 121, column: 27, scope: !7)
!34 = !DILocation(line: 121, column: 40, scope: !7)
!35 = !DILocation(line: 110, column: 4, scope: !7)
